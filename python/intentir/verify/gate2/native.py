"""CUDA and TileLang execution adapters for the three-way Gate2 protocol."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompiler,
    CudaKernelSource,
)
from intentir.verify.gate1.tilelang_compiler import TileLangKernelSource

from .model import ExecutionResult


class NativeArgumentRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    SCALAR = "scalar"


@dataclass(frozen=True)
class NativeArgument:
    role: NativeArgumentRole
    value_key: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    value: int | float | bool | None = None
    output_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, NativeArgumentRole):
            object.__setattr__(self, "role", NativeArgumentRole(self.role))
        if self.role in {NativeArgumentRole.INPUT, NativeArgumentRole.INOUT} and not self.value_key:
            raise ValueError(f"{self.role.value} native argument requires value_key")
        if self.role is NativeArgumentRole.OUTPUT:
            if self.shape is None or self.dtype is None:
                raise ValueError("native output requires shape and dtype")
            if self.output_key is None:
                raise ValueError("native output requires an observable output_key")
        if self.role is NativeArgumentRole.SCALAR and self.value_key is None and self.value is None:
            raise ValueError("native scalar requires value_key or fixed value")


@dataclass(frozen=True)
class NativeExecutionSpec:
    arguments: tuple[NativeArgument, ...]
    compile_case: Any
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem: int = 0
    device: str = "cuda"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("native execution requires ordered arguments")
        for label, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"native {label} must contain three positive dimensions")
            object.__setattr__(self, label, dims)
        if not isinstance(self.shared_mem, int) or self.shared_mem < 0:
            raise ValueError("native shared memory must be a non-negative integer")


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
    }
    key = aliases.get(dtype.removeprefix("torch.").lower(), dtype.removeprefix("torch.").lower())
    try:
        return getattr(torch, key)
    except AttributeError as exc:
        raise ValueError(f"unsupported native tensor dtype {dtype!r}") from exc


def _scalar_ctype(dtype: str | None, value: Any) -> type[Any]:
    key = str(dtype or "").removeprefix("torch.").lower()
    mapping: dict[str, type[Any]] = {
        "bool": ctypes.c_bool,
        "f32": ctypes.c_float,
        "float32": ctypes.c_float,
        "f64": ctypes.c_double,
        "float64": ctypes.c_double,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
        "u8": ctypes.c_uint8,
        "u16": ctypes.c_uint16,
        "u32": ctypes.c_uint32,
        "u64": ctypes.c_uint64,
    }
    if key in mapping:
        return mapping[key]
    if isinstance(value, bool):
        return ctypes.c_bool
    if isinstance(value, int):
        return ctypes.c_int32
    if isinstance(value, float):
        return ctypes.c_float
    raise ValueError(f"unsupported native scalar dtype {dtype!r}")


def _prepare_torch_arguments(
    spec: NativeExecutionSpec,
    values: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    import torch

    arguments: list[Any] = []
    outputs: dict[str, Any] = {}
    for index, argument in enumerate(spec.arguments):
        if argument.role is NativeArgumentRole.SCALAR:
            value = values[argument.value_key] if argument.value_key is not None else argument.value
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            arguments.append(value)
            continue
        dtype = _torch_dtype(torch, argument.dtype or "f32")
        if argument.role is NativeArgumentRole.OUTPUT:
            tensor = torch.empty(tuple(argument.shape or ()), dtype=dtype, device=spec.device)
            outputs[argument.output_key or f"output_{index}"] = tensor
        else:
            tensor = torch.as_tensor(
                np.asarray(values[argument.value_key]), dtype=dtype, device=spec.device
            )
            if argument.role is NativeArgumentRole.INOUT:
                tensor = tensor.clone()
                outputs[argument.output_key or f"output_{index}"] = tensor
        arguments.append(tensor)
    return arguments, outputs


def _copy_outputs(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: tensor.detach().cpu().numpy().copy()
        for key, tensor in outputs.items()
    }


def _driver_check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA driver {operation} failed with {error}")
    return result[1:]


class CudaRunner:
    """Compile to PTX and launch through the CUDA driver on isolated tensors."""

    def __init__(self, compiler: CudaCompiler | None = None) -> None:
        self.compiler = compiler or CudaCompiler()

    def run(
        self,
        leg: str,
        program: CudaKernelSource,
        spec: NativeExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        module = None
        try:
            import torch
            from cuda.bindings import driver

            if not isinstance(spec.compile_case, CudaCompileCase):
                raise TypeError("CUDA Gate2 requires a CudaCompileCase")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            compiled = self.compiler.compile(program, spec.compile_case)
            if not compiled.ok or compiled.ptx is None or compiled.entry is None:
                raise RuntimeError(compiled.diagnostic or "CUDA Gate2 compile failed")
            device = torch.device(spec.device)
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(device_index)
            # Materializing one tensor ensures PyTorch has retained the primary context.
            torch.empty((), device=spec.device)
            _driver_check(driver.cuInit(0), "cuInit")
            module, = _driver_check(
                driver.cuModuleLoadData(compiled.ptx.encode("utf-8")),
                "cuModuleLoadData",
            )
            function, = _driver_check(
                driver.cuModuleGetFunction(module, compiled.entry.encode("utf-8")),
                "cuModuleGetFunction",
            )
            arguments, outputs = _prepare_torch_arguments(spec, values)
            packed_values: list[Any] = []
            packed_types: list[type[Any]] = []
            for definition, value in zip(spec.arguments, arguments, strict=True):
                if definition.role is NativeArgumentRole.SCALAR:
                    ctype = _scalar_ctype(definition.dtype, value)
                    packed_values.append(ctype(value))
                    packed_types.append(ctype)
                else:
                    packed_values.append(ctypes.c_void_p(value.data_ptr()))
                    packed_types.append(ctypes.c_void_p)
            _driver_check(
                driver.cuLaunchKernel(
                    function,
                    *spec.grid,
                    *spec.block,
                    spec.shared_mem,
                    0,
                    (tuple(packed_values), tuple(packed_types)),
                    0,
                ),
                "cuLaunchKernel",
            )
            torch.cuda.synchronize(device_index)
            return ExecutionResult(
                leg,
                True,
                _copy_outputs(outputs),
                metadata={
                    "frontend": "cuda",
                    "entry": compiled.entry,
                    "grid": list(spec.grid),
                    "block": list(spec.block),
                    **dict(spec.metadata),
                },
            )
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if module is not None:
                try:
                    from cuda.bindings import driver

                    driver.cuModuleUnload(module)
                except Exception:
                    pass


def _load_tilelang_prim_func(program: TileLangKernelSource, temporary: Path) -> tuple[Any, str | None]:
    if program.kind == "factory_file":
        assert program.path is not None
        path = Path(program.path).resolve()
    else:
        assert program.code is not None
        path = temporary / "replay.py"
        path.write_text(program.code, encoding="utf-8")
    module_name = f"_intentir_gate2_tilelang_{id(program):x}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load TileLang module {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
        value = getattr(module, program.function)
        if program.kind == "factory_file":
            value = value(**dict(program.factory_kwargs))
        return value, module_name
    except Exception:
        sys.modules.pop(module_name, None)
        raise


class TileLangRunner:
    """Compile a PrimFunc to a callable JITKernel and execute ordered tensors."""

    def run(
        self,
        leg: str,
        program: TileLangKernelSource,
        spec: NativeExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        module_name: str | None = None
        try:
            import tilelang
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            with tempfile.TemporaryDirectory(prefix=f"intentir-tilelang-gate2-{leg}-") as raw:
                prim_func, module_name = _load_tilelang_prim_func(program, Path(raw))
                target = str(getattr(spec.compile_case, "target", "cuda"))
                kernel = tilelang.compile(prim_func, target=target)
                arguments, outputs = _prepare_torch_arguments(spec, values)
                kernel(*arguments)
                torch.cuda.synchronize(spec.device)
                return ExecutionResult(
                    leg,
                    True,
                    _copy_outputs(outputs),
                    metadata={
                        "frontend": "tilelang",
                        "grid": list(spec.grid),
                        "block": list(spec.block),
                        **dict(spec.metadata),
                    },
                )
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if module_name is not None:
                sys.modules.pop(module_name, None)


__all__ = [
    "CudaRunner",
    "NativeArgument",
    "NativeArgumentRole",
    "NativeExecutionSpec",
    "TileLangRunner",
]
