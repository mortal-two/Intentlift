"""Actual final-J replay Triton execution adapter for Gate2."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.triton_compat import normalize_deprecated_triton_dot

from .model import ExecutionResult


class ArgumentRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    SCALAR = "scalar"
    NONE = "none"


@dataclass(frozen=True)
class TritonArgument:
    name: str
    role: ArgumentRole
    value_key: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    value: int | float | bool | None = None
    output_key: str | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid Triton argument name: {self.name!r}")
        if not isinstance(self.role, ArgumentRole):
            object.__setattr__(self, "role", ArgumentRole(self.role))
        if self.role in {ArgumentRole.INPUT, ArgumentRole.INOUT} and not self.value_key:
            raise ValueError(f"{self.role.value} argument requires value_key")
        if self.role is ArgumentRole.OUTPUT:
            if self.shape is None or self.dtype is None:
                raise ValueError("output argument requires shape and dtype")
            if not self.output_key:
                object.__setattr__(self, "output_key", self.name)
        if self.strides is not None:
            strides = tuple(int(value) for value in self.strides)
            if self.shape is not None and len(strides) != len(self.shape):
                raise ValueError("Triton argument strides must match its rank")
            if any(value < 0 for value in strides):
                raise ValueError("negative Triton strides require a storage offset")
            object.__setattr__(self, "strides", strides)
        if self.role is ArgumentRole.SCALAR and self.value_key is None and self.value is None:
            raise ValueError("scalar argument requires value_key or fixed value")
        if self.role is ArgumentRole.NONE and any(
            item is not None
            for item in (
                self.value_key,
                self.dtype,
                self.shape,
                self.value,
                self.output_key,
                self.strides,
            )
        ):
            raise ValueError("none argument forbids value/dtype/shape/output fields")


def _storage_size(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if len(shape) != len(strides):
        raise ValueError("physical tensor shape/stride ranks differ")
    if not shape:
        return 1
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))


def _physical_offsets(shape: tuple[int, ...], strides: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return np.asarray(0, dtype=np.int64)
    offsets = np.zeros(shape, dtype=np.int64)
    for axis, (extent, stride) in enumerate(zip(shape, strides, strict=True)):
        axis_values = np.arange(extent, dtype=np.int64)
        axis_values = axis_values.reshape((1,) * axis + (extent,) + (1,) * (len(shape) - axis - 1))
        offsets += axis_values * stride
    return offsets


def _input_tensor(
    torch: Any,
    raw: np.ndarray,
    *,
    dtype: Any,
    device: str,
    strides: tuple[int, ...] | None,
) -> Any:
    if strides is None:
        return torch.as_tensor(raw, dtype=dtype, device=device)
    shape = tuple(int(value) for value in raw.shape)
    offsets = _physical_offsets(shape, strides)
    storage = np.zeros(_storage_size(shape, strides), dtype=raw.dtype)
    storage[offsets.reshape(-1)] = raw.reshape(-1)
    if not np.array_equal(storage[offsets], raw, equal_nan=True):
        raise ValueError("logical input values conflict under captured overlapping strides")
    base = torch.as_tensor(storage, dtype=dtype, device=device)
    return torch.as_strided(base, size=shape, stride=strides)


def _output_tensor(
    torch: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: str,
    strides: tuple[int, ...] | None,
) -> Any:
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)
    base = torch.empty(_storage_size(shape, strides), dtype=dtype, device=device)
    return torch.as_strided(base, size=shape, stride=strides)


@dataclass(frozen=True)
class TritonExecutionSpec:
    arguments: tuple[TritonArgument, ...]
    grid: tuple[int, ...]
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("Triton execution spec requires arguments")
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("duplicate Triton argument names")
        schedule = dict(self.schedule)
        invalid_schedule_names = sorted(
            repr(name)
            for name in schedule
            if not isinstance(name, str) or not name.isidentifier()
        )
        if invalid_schedule_names:
            raise ValueError(
                "invalid Triton schedule names: "
                + ", ".join(invalid_schedule_names)
            )
        overlap = sorted(set(names) & set(schedule))
        if overlap:
            raise ValueError(
                "Triton arguments and schedule overlap: " + ", ".join(overlap)
            )
        object.__setattr__(self, "schedule", schedule)
        if not 1 <= len(self.grid) <= 3 or any(int(dim) <= 0 for dim in self.grid):
            raise ValueError("grid must contain one to three positive dimensions")


def _torch_dtype(torch: Any, dtype: str) -> Any:
    key = dtype.removeprefix("torch.").lower()
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "i1": "bool",
    }
    key = aliases.get(key, key)
    try:
        return getattr(torch, key)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype {dtype!r}") from exc


def _load_module(
    source: KernelSource, temporary: Path
) -> tuple[ModuleType, str, dict[str, Any]]:
    code, origin = source.read_code()
    normalized, frontend_compat = normalize_deprecated_triton_dot(code)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    module_name = f"_intentir_gate2_{source.function}_{digest}_{id(source):x}"
    if source.kind == "module_file" and not frontend_compat["rewrite_count"]:
        assert source.path is not None
        path = Path(source.path).resolve()
    else:
        path = temporary / f"{module_name}.py"
        path.write_text(normalized, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Triton module from {origin}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, module_name, frontend_compat


def _launch_by_name(
    kernel: Any,
    grid: tuple[int, ...],
    runtime_arguments: Mapping[str, Any],
    schedule: Mapping[str, int | float | bool | str],
) -> None:
    """Bind a complete Triton ABI by name, including interleaved constexprs."""

    runtime = dict(runtime_arguments)
    frozen_schedule = dict(schedule)
    overlap = sorted(set(runtime) & set(frozen_schedule))
    if overlap:
        raise ValueError(
            "Triton runtime arguments and schedule overlap: "
            + ", ".join(overlap)
        )
    declared = getattr(kernel, "arg_names", None)
    if not isinstance(declared, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in declared
    ):
        raise TypeError("Triton replay kernel has no valid arg_names interface")
    declared_names = list(declared)
    if len(declared_names) != len(set(declared_names)):
        raise ValueError("Triton replay kernel has duplicate arg_names")
    unknown_runtime = sorted(set(runtime) - set(declared_names))
    if unknown_runtime:
        raise ValueError(
            "runtime arguments not in Triton replay interface: "
            + ", ".join(unknown_runtime)
        )
    missing = sorted(
        set(declared_names) - set(runtime) - set(frozen_schedule)
    )
    if missing:
        raise ValueError(
            "missing Triton replay interface arguments: " + ", ".join(missing)
        )
    # Triton may canonicalize an interleaved Python signature by placing
    # runtime arguments before constexpr arguments.  Launches below are fully
    # name-keyed, so dictionary/declaration order is not part of the ABI.  The
    # completeness, overlap, and unknown-name checks above remain authoritative.
    launch_kwargs = {**runtime, **frozen_schedule}
    kernel[grid](**launch_kwargs)


class TritonRunner:
    """Load and launch the final-J replay executable independently."""

    def run(
        self,
        leg: str,
        program: KernelSource,
        spec: TritonExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        try:
            import torch
            import triton  # noqa: F401

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            with tempfile.TemporaryDirectory(prefix=f"intentir-gate2-{leg}-") as tmp:
                module, module_name, frontend_compat = _load_module(
                    program, Path(tmp)
                )
                try:
                    kernel = getattr(module, program.function)
                    launch_args: dict[str, Any] = {}
                    outputs: dict[str, Any] = {}
                    for argument in spec.arguments:
                        if argument.role is ArgumentRole.NONE:
                            launch_args[argument.name] = None
                            continue
                        if argument.role is ArgumentRole.SCALAR:
                            scalar = (
                                values[argument.value_key]
                                if argument.value_key is not None
                                else argument.value
                            )
                            if isinstance(scalar, np.generic):
                                scalar = scalar.item()
                            launch_args[argument.name] = scalar
                            continue
                        dtype = _torch_dtype(torch, argument.dtype or "float32")
                        if argument.role is ArgumentRole.OUTPUT:
                            tensor = _output_tensor(
                                torch,
                                shape=tuple(argument.shape or ()),
                                dtype=dtype,
                                device=spec.device,
                                strides=argument.strides,
                            )
                            outputs[argument.output_key or argument.name] = tensor
                        else:
                            raw = np.asarray(values[argument.value_key])
                            tensor = _input_tensor(
                                torch,
                                raw,
                                dtype=dtype,
                                device=spec.device,
                                strides=argument.strides,
                            )
                            if argument.role is ArgumentRole.INOUT:
                                outputs[argument.output_key or argument.name] = tensor
                        launch_args[argument.name] = tensor
                    _launch_by_name(
                        kernel,
                        tuple(int(dim) for dim in spec.grid),
                        launch_args,
                        spec.schedule,
                    )
                    torch.cuda.synchronize(spec.device)
                    copied = {
                        key: tensor.detach().cpu().numpy().copy()
                        for key, tensor in outputs.items()
                    }
                    return ExecutionResult(
                        leg,
                        True,
                        copied,
                        metadata={
                            "function": program.function,
                            "grid": list(spec.grid),
                            "binding_protocol": (
                                "triton_name_keyed_complete_abi_v1"
                            ),
                            "frontend_compat": frontend_compat,
                        },
                    )
                finally:
                    sys.modules.pop(module_name, None)
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")


__all__ = [
    "ArgumentRole",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
]
