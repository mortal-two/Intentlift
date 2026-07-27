"""Versioned natural-workload bindings for uncertified backend J variants.

The compiler never reads this module.  It belongs to the benchmark harness and
turns an existing real Source adapter into an exact shape/dtype invocation.
Each configuration pins the digest of the base adapter, so upstream adapter
changes cannot silently alter a published workload.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from intentir.ir.production_snapshot import canonical_json, content_digest
from intentir.verify.workload_adapters import (
    PreparedTritonWorkload,
    TRITON_WORKLOAD_ADAPTERS,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)


DEV_WORKLOAD_PROTOCOL = "intentir.backend.dev-workload.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")
_DTYPES = frozenset(
    {"i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"}
)


class DevWorkloadError(ValueError):
    """Raised when a development workload is incomplete or stale."""


def _strict_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise DevWorkloadError(f"{name} must be an object with non-empty string keys")
    try:
        return json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise DevWorkloadError(f"{name} is not strict JSON: {exc}") from exc


def _stable_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DevWorkloadError(f"{name} must be a stable non-empty identifier")


@dataclass(frozen=True)
class DevWorkloadConfig:
    case_id: str
    shape_config_id: str
    dtype_config_id: str
    base_adapter_digest: str
    concrete_extents: Mapping[str, int]
    buffer_shapes: Mapping[str, tuple[int, ...]]
    buffer_dtypes: Mapping[str, str]
    scalar_values: Mapping[str, bool | int | float]
    semantic_constexprs: Mapping[str, Any]
    c_values: Mapping[str, Any]
    grid: tuple[int, ...]
    grid_contract: tuple[str, ...]
    upstream_evidence: tuple[str, ...]
    protocol: str = DEV_WORKLOAD_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != DEV_WORKLOAD_PROTOCOL:
            raise DevWorkloadError("unsupported development workload protocol")
        for name in ("case_id", "shape_config_id", "dtype_config_id"):
            _stable_id(name, getattr(self, name))
        if (
            not isinstance(self.base_adapter_digest, str)
            or _SHA256.fullmatch(self.base_adapter_digest) is None
        ):
            raise DevWorkloadError("base_adapter_digest must be SHA-256")
        extents = _strict_mapping("concrete_extents", self.concrete_extents)
        if not extents or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in extents.values()
        ):
            raise DevWorkloadError("concrete_extents must contain positive integers")
        raw_shapes = _strict_mapping("buffer_shapes", self.buffer_shapes)
        shapes: dict[str, tuple[int, ...]] = {}
        for name, shape in raw_shapes.items():
            if not isinstance(shape, list) or not shape or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in shape
            ):
                raise DevWorkloadError(
                    f"development buffer shape {name!r} must be positive integers"
                )
            shapes[name] = tuple(shape)
        dtypes = _strict_mapping("buffer_dtypes", self.buffer_dtypes)
        if any(value not in _DTYPES for value in dtypes.values()):
            raise DevWorkloadError("development workload uses an unknown buffer dtype")
        scalar_values = _strict_mapping("scalar_values", self.scalar_values)
        if any(not isinstance(value, (bool, int, float)) for value in scalar_values.values()):
            raise DevWorkloadError("development scalar values must be bool/int/float")
        semantics = _strict_mapping(
            "semantic_constexprs", self.semantic_constexprs
        )
        c_values = _strict_mapping("c_values", self.c_values)
        grid = tuple(self.grid)
        if not 1 <= len(grid) <= 3 or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in grid
        ):
            raise DevWorkloadError("development workload grid must be positive")
        grid_contract = tuple(self.grid_contract)
        if len(grid_contract) != len(grid) or any(
            not isinstance(item, str) or not item for item in grid_contract
        ):
            raise DevWorkloadError("grid_contract must describe every grid dimension")
        if (
            not isinstance(self.upstream_evidence, tuple)
            or not self.upstream_evidence
            or any(not isinstance(item, str) or not item for item in self.upstream_evidence)
        ):
            raise DevWorkloadError("development workload requires upstream evidence")
        if not any((shapes, dtypes, scalar_values, semantics, c_values)):
            raise DevWorkloadError("development workload must change at least one binding")
        object.__setattr__(self, "concrete_extents", extents)
        object.__setattr__(self, "buffer_shapes", shapes)
        object.__setattr__(self, "buffer_dtypes", dtypes)
        object.__setattr__(self, "scalar_values", scalar_values)
        object.__setattr__(self, "semantic_constexprs", semantics)
        object.__setattr__(self, "c_values", c_values)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_contract", grid_contract)

    def _content(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "case_id": self.case_id,
            "shape_config_id": self.shape_config_id,
            "dtype_config_id": self.dtype_config_id,
            "base_adapter_digest": self.base_adapter_digest,
            "concrete_extents": dict(self.concrete_extents),
            "buffer_shapes": {
                name: list(shape) for name, shape in self.buffer_shapes.items()
            },
            "buffer_dtypes": dict(self.buffer_dtypes),
            "scalar_values": dict(self.scalar_values),
            "semantic_constexprs": dict(self.semantic_constexprs),
            "c_values": dict(self.c_values),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "upstream_evidence": list(self.upstream_evidence),
        }

    @property
    def config_digest(self) -> str:
        return content_digest(self._content())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "config_digest": self.config_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DevWorkloadConfig":
        expected = {
            "protocol",
            "case_id",
            "shape_config_id",
            "dtype_config_id",
            "base_adapter_digest",
            "concrete_extents",
            "buffer_shapes",
            "buffer_dtypes",
            "scalar_values",
            "semantic_constexprs",
            "c_values",
            "grid",
            "grid_contract",
            "upstream_evidence",
            "config_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise DevWorkloadError(
                f"development workload must contain exactly {sorted(expected)}"
            )
        for field in ("grid", "grid_contract", "upstream_evidence"):
            if not isinstance(value[field], list):
                raise DevWorkloadError(f"development workload {field} must be a list")
        config = cls(
            protocol=value["protocol"],
            case_id=value["case_id"],
            shape_config_id=value["shape_config_id"],
            dtype_config_id=value["dtype_config_id"],
            base_adapter_digest=value["base_adapter_digest"],
            concrete_extents=value["concrete_extents"],
            buffer_shapes=value["buffer_shapes"],
            buffer_dtypes=value["buffer_dtypes"],
            scalar_values=value["scalar_values"],
            semantic_constexprs=value["semantic_constexprs"],
            c_values=value["c_values"],
            grid=tuple(value["grid"]),
            grid_contract=tuple(value["grid_contract"]),
            upstream_evidence=tuple(value["upstream_evidence"]),
        )
        if value["config_digest"] != config.config_digest:
            raise DevWorkloadError("development workload config_digest mismatch")
        return config

    @classmethod
    def from_json(cls, payload: str) -> "DevWorkloadConfig":
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DevWorkloadError(f"invalid development workload JSON: {exc}") from exc
        return cls.from_dict(value)


def apply_dev_workload_config(
    adapter: TritonWorkloadAdapter,
    config: DevWorkloadConfig,
) -> TritonWorkloadAdapter:
    """Apply one digest-pinned harness configuration without compiler policy."""

    if not isinstance(adapter, TritonWorkloadAdapter):
        raise DevWorkloadError("development workload requires TritonWorkloadAdapter")
    if not isinstance(config, DevWorkloadConfig):
        raise DevWorkloadError("development workload requires DevWorkloadConfig")
    if adapter.case_id != config.case_id:
        raise DevWorkloadError("development workload case_id mismatch")
    observed_digest = content_digest(adapter.to_json())
    if observed_digest != config.base_adapter_digest:
        raise DevWorkloadError(
            "development workload base adapter changed: expected "
            f"{config.base_adapter_digest}, observed {observed_digest}"
        )

    buffers_by_name = {item.formal: item for item in adapter.buffers}
    unknown_buffers = (
        set(config.buffer_shapes) | set(config.buffer_dtypes)
    ) - set(buffers_by_name)
    if unknown_buffers:
        raise DevWorkloadError(
            f"development workload names unknown buffers: {sorted(unknown_buffers)}"
        )
    buffers = tuple(
        replace(
            item,
            shape=config.buffer_shapes.get(item.formal, item.shape),
            dtype=config.buffer_dtypes.get(item.formal, item.dtype),
        )
        for item in adapter.buffers
    )

    scalar_names = {item.formal for item in adapter.scalars}
    unknown_scalars = set(config.scalar_values) - scalar_names
    if unknown_scalars:
        raise DevWorkloadError(
            f"development workload names unknown scalars: {sorted(unknown_scalars)}"
        )
    scalars = tuple(
        replace(
            item,
            value=config.scalar_values.get(item.formal, item.value),
        )
        for item in adapter.scalars
    )

    def update_pairs(
        label: str,
        original: tuple[tuple[str, Any], ...],
        updates: Mapping[str, Any],
    ) -> tuple[tuple[str, Any], ...]:
        names = {name for name, _value in original}
        unknown = set(updates) - names
        if unknown:
            raise DevWorkloadError(
                f"development workload names unknown {label}: {sorted(unknown)}"
            )
        return tuple((name, updates.get(name, value)) for name, value in original)

    semantics = update_pairs(
        "semantic constexprs",
        adapter.semantic_constexprs,
        config.semantic_constexprs,
    )
    c_values = update_pairs("C values", adapter.c_values, config.c_values)
    active = replace(
        adapter,
        buffers=buffers,
        scalars=scalars,
        semantic_constexprs=semantics,
        c_values=c_values,
        grid=config.grid,
        grid_contract=config.grid_contract,
    )
    if content_digest(active.to_json()) == observed_digest:
        raise DevWorkloadError("development workload did not change its base adapter")
    return active


def load_dev_workload_config(path: str | Path) -> DevWorkloadConfig:
    target = Path(path)
    try:
        return DevWorkloadConfig.from_json(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DevWorkloadError(
            f"cannot read development workload config {target}: {exc}"
        ) from exc


def prepare_dev_workload(config: DevWorkloadConfig) -> PreparedTritonWorkload:
    """Resolve the real Source after applying a digest-pinned dev invocation."""

    if not isinstance(config, DevWorkloadConfig):
        raise DevWorkloadError("prepare_dev_workload requires DevWorkloadConfig")
    adapter = TRITON_WORKLOAD_ADAPTERS.get(config.case_id)
    if adapter is None:
        raise DevWorkloadError(
            f"development workload has no real Source adapter for {config.case_id!r}"
        )
    return prepare_triton_workload(apply_dev_workload_config(adapter, config))


__all__ = [
    "DEV_WORKLOAD_PROTOCOL",
    "DevWorkloadConfig",
    "DevWorkloadError",
    "apply_dev_workload_config",
    "load_dev_workload_config",
    "prepare_dev_workload",
]
