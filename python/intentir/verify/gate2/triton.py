"""Actual source/replay Triton execution legs for Gate2."""

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

from .model import ExecutionResult


class ArgumentRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    SCALAR = "scalar"


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


@dataclass(frozen=True)
class TritonExecutionSpec:
    arguments: tuple[TritonArgument, ...]
    grid: tuple[int, ...]
    schedule: Mapping[str, int | float | bool] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("Triton execution spec requires arguments")
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("duplicate Triton argument names")
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


class TritonRunner:
    """Load and launch one Triton program; source and replay use separate calls."""

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
                module, module_name = _load_module(program, Path(tmp))
                try:
                    kernel = getattr(module, program.function)
                    launch_args: list[Any] = []
                    outputs: dict[str, Any] = {}
                    for argument in spec.arguments:
                        if argument.role is ArgumentRole.SCALAR:
                            scalar = (
                                values[argument.value_key]
                                if argument.value_key is not None
                                else argument.value
                            )
                            if isinstance(scalar, np.generic):
                                scalar = scalar.item()
                            launch_args.append(scalar)
                            continue
                        dtype = _torch_dtype(torch, argument.dtype or "float32")
                        if argument.role is ArgumentRole.OUTPUT:
                            tensor = torch.empty(
                                tuple(argument.shape or ()), dtype=dtype, device=spec.device
                            )
                            outputs[argument.output_key or argument.name] = tensor
                        else:
                            raw = np.asarray(values[argument.value_key])
                            tensor = torch.as_tensor(raw, dtype=dtype, device=spec.device)
                            if argument.role is ArgumentRole.INOUT:
                                tensor = tensor.clone()
                                outputs[argument.output_key or argument.name] = tensor
                        launch_args.append(tensor)
                    kernel[tuple(int(dim) for dim in spec.grid)](
                        *launch_args, **dict(spec.schedule)
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
                        metadata={"function": program.function, "grid": list(spec.grid)},
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
