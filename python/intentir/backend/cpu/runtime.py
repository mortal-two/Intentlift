"""Crash-isolated runtime for the scalar CPU ranked-memref entry point."""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .abi import CPU_C_INTERFACE_SYMBOL, descriptor_for

if TYPE_CHECKING:
    from .compiler import CompiledCPUModule


CPU_RUNTIME_PROTOCOL = "intentir_cpu_isolated_runtime_v1"


class CPURuntimeError(RuntimeError):
    """The compiled scalar module could not be executed safely."""


@dataclass(frozen=True)
class CPURuntimeResult:
    outputs: tuple[np.ndarray, ...]
    metadata: Mapping[str, Any]


def _native_call(
    shared_object: Path,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
) -> None:
    """Load and invoke the single C-interface entry. Used only by the worker."""

    try:
        library = ctypes.CDLL(str(shared_object), mode=ctypes.RTLD_LOCAL)
        entry = getattr(library, CPU_C_INTERFACE_SYMBOL)
    except (OSError, AttributeError) as exc:
        raise CPURuntimeError(
            f"could not load scalar CPU entry {CPU_C_INTERFACE_SYMBOL!r}: {exc}"
        ) from exc

    arrays = [*inputs, *outputs]
    descriptors = [descriptor_for(value) for value in arrays]
    entry.argtypes = [ctypes.POINTER(type(item)) for item in descriptors]
    entry.restype = None
    entry(*(ctypes.byref(item) for item in descriptors))


def _runtime_timeout_seconds() -> float:
    spelling = os.environ.get("INTENTIR_CPU_RUNTIME_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(spelling)
    except ValueError as exc:
        raise CPURuntimeError(
            "INTENTIR_CPU_RUNTIME_TIMEOUT_SECONDS must be numeric"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise CPURuntimeError(
            "INTENTIR_CPU_RUNTIME_TIMEOUT_SECONDS must be finite and positive"
        )
    return timeout


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def execute_isolated(
    compiled: "CompiledCPUModule",
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
) -> CPURuntimeResult:
    """Execute exactly one native entry in a subprocess and copy results back.

    File-backed NumPy mappings keep the verifier process safe from generated
    code crashes and avoid serializing tensor values through command arguments
    or JSON.  A signal, timeout, or worker exception is an execution failure;
    no interpreter or target-backend fallback is attempted.
    """

    with tempfile.TemporaryDirectory(prefix="intentir-cpu-run-") as directory:
        root = Path(directory)
        input_paths: list[Path] = []
        output_paths: list[Path] = []

        for index, value in enumerate(inputs):
            path = root / f"input-{index}.npy"
            mapped = np.lib.format.open_memmap(
                path, mode="w+", dtype=value.dtype, shape=value.shape
            )
            np.copyto(mapped, value, casting="no")
            mapped.flush()
            _close_memmap(mapped)
            input_paths.append(path)

        for index, value in enumerate(outputs):
            path = root / f"output-{index}.npy"
            mapped = np.lib.format.open_memmap(
                path, mode="w+", dtype=value.dtype, shape=value.shape
            )
            mapped.flush()
            _close_memmap(mapped)
            output_paths.append(path)

        request = {
            "protocol": CPU_RUNTIME_PROTOCOL,
            "shared_object": str(compiled.shared_object),
            "inputs": [str(path) for path in input_paths],
            "outputs": [str(path) for path in output_paths],
        }
        environment = dict(os.environ)
        python_root = str(Path(__file__).resolve().parents[3])
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            python_root if not inherited else os.pathsep.join((python_root, inherited))
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "intentir.backend.cpu.worker"],
                input=json.dumps(request, separators=(",", ":")),
                capture_output=True,
                text=True,
                check=False,
                timeout=_runtime_timeout_seconds(),
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise CPURuntimeError(
                "scalar CPU worker exceeded the configured runtime timeout"
            ) from exc

        if completed.returncode:
            if completed.returncode < 0:
                detail = f"terminated by signal {-completed.returncode}"
            else:
                detail = f"exited with status {completed.returncode}"
            diagnostic = (completed.stderr or completed.stdout).strip()
            if diagnostic:
                detail += f": {diagnostic[:1000]}"
            raise CPURuntimeError(f"scalar CPU worker {detail}")

        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CPURuntimeError(
                "scalar CPU worker returned malformed execution evidence"
            ) from exc
        if not isinstance(response, dict):
            raise CPURuntimeError(
                "scalar CPU worker returned a non-object execution response"
            )
        if response != {"native_call_count": 1, "ok": True}:
            raise CPURuntimeError(
                "scalar CPU worker rejected execution: "
                f"{response.get('error', response)!r}"
            )

        result: list[np.ndarray] = []
        for expected, path in zip(outputs, output_paths, strict=True):
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                value = np.array(mapped, copy=True, order="C")
            finally:
                _close_memmap(mapped)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise CPURuntimeError(
                    "scalar CPU worker changed an output ABI contract"
                )
            result.append(value)

    return CPURuntimeResult(
        tuple(result),
        {
            "runtime_protocol": CPU_RUNTIME_PROTOCOL,
            "entry_symbol": CPU_C_INTERFACE_SYMBOL,
            "native_call_count": 1,
            "isolated_process": True,
        },
    )


__all__ = [
    "CPU_RUNTIME_PROTOCOL",
    "CPURuntimeError",
    "CPURuntimeResult",
    "execute_isolated",
]
