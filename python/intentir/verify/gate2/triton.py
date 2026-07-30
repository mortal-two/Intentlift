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

from intentir.runtime_dtype import torch_tensor_from_numpy, torch_tensor_to_numpy
from intentir.verify.gate1.compiler import KernelSource

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
            )
        ):
            raise ValueError("none argument forbids value/dtype/shape/output fields")


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


def _load_module(source: KernelSource, temporary: Path) -> tuple[ModuleType, str]:
    code, origin = source.read_code()
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    module_name = f"_intentir_gate2_{source.function}_{digest}_{id(source):x}"
    if source.kind == "module_file":
        assert source.path is not None
        path = Path(source.path).resolve()
    else:
        path = temporary / f"{module_name}.py"
        path.write_text(code, encoding="utf-8")
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
    return module, module_name


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
    expected_runtime_order = [
        name for name in declared_names if name in runtime
    ]
    if list(runtime) != expected_runtime_order:
        raise ValueError(
            "runtime argument order differs from the Triton replay ABI: "
            f"{list(runtime)!r} != {expected_runtime_order!r}"
        )
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
            with PreparedTritonInvocation(program, spec, values) as invocation:
                invocation.launch()
                torch.cuda.synchronize(spec.device)
                return ExecutionResult(
                    leg,
                    True,
                    invocation.copied_outputs(),
                    metadata={
                        "function": program.function,
                        "grid": list(spec.grid),
                        "binding_protocol": (
                            "triton_name_keyed_complete_abi_v1"
                        ),
                    },
                )
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")


class PreparedTritonInvocation:
    """Load once, bind once, then expose a no-sync Triton launch closure.

    Gate2 uses one launch followed by an explicit synchronization. Performance
    reporting reuses the identical source/spec/value binding but times only
    repeated kernel launches. Compilation, imports, allocations and host-device
    copies remain outside the timed closure.
    """

    def __init__(
        self,
        program: KernelSource,
        spec: TritonExecutionSpec,
        values: Mapping[str, Any],
    ) -> None:
        import torch

        self._temporary = tempfile.TemporaryDirectory(
            prefix="intentir-triton-prepared-"
        )
        module, self._module_name = _load_module(
            program, Path(self._temporary.name)
        )
        self._kernel = getattr(module, program.function)
        self._grid = tuple(int(dim) for dim in spec.grid)
        self._schedule = dict(spec.schedule)
        self._device = spec.device
        self._launch_args: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self._mutable_input_resets: list[tuple[Any, Any]] = []
        for argument in spec.arguments:
            if argument.role is ArgumentRole.NONE:
                self._launch_args[argument.name] = None
                continue
            if argument.role is ArgumentRole.SCALAR:
                scalar = (
                    values[argument.value_key]
                    if argument.value_key is not None
                    else argument.value
                )
                if isinstance(scalar, np.ndarray) and scalar.size == 1:
                    scalar = scalar.item()
                elif isinstance(scalar, np.generic):
                    scalar = scalar.item()
                self._launch_args[argument.name] = scalar
                continue
            dtype = _torch_dtype(torch, argument.dtype or "float32")
            if argument.role is ArgumentRole.OUTPUT:
                tensor = torch.empty(
                    tuple(argument.shape or ()),
                    dtype=dtype,
                    device=spec.device,
                )
                self.outputs[argument.output_key or argument.name] = tensor
            else:
                raw = np.asarray(values[argument.value_key])
                tensor = torch_tensor_from_numpy(
                    torch,
                    raw,
                    dtype=dtype,
                    device=spec.device,
                )
                if argument.role is ArgumentRole.INOUT:
                    tensor = tensor.clone()
                    self.outputs[argument.output_key or argument.name] = tensor
                    self._mutable_input_resets.append((tensor, tensor.clone()))
            self._launch_args[argument.name] = tensor

    def launch(self) -> None:
        _launch_by_name(
            self._kernel,
            self._grid,
            self._launch_args,
            self._schedule,
        )

    def thunk(self):
        return self.launch

    def reset_mutable_inputs(self) -> None:
        """Restore INOUT arguments to the authored workload valuation."""

        for destination, pristine in self._mutable_input_resets:
            destination.copy_(pristine)

    def copied_outputs(self) -> dict[str, np.ndarray]:
        import torch

        return {
            key: torch_tensor_to_numpy(torch, tensor)
            for key, tensor in self.outputs.items()
        }

    def close(self) -> None:
        module_name = getattr(self, "_module_name", None)
        if module_name is not None:
            sys.modules.pop(module_name, None)
            self._module_name = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "PreparedTritonInvocation":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "ArgumentRole",
    "PreparedTritonInvocation",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
]
