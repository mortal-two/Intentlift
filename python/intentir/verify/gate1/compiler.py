"""Isolated compile-only Triton frontend for SCRC Gate1.

The public classes in this module are intentionally independent of the rest of
the Gate1 model/orchestrator.  They form a small, JSON-serializable boundary:
the parent process validates replay Python, describes compile-time arguments,
and starts a fresh worker process; the worker imports Triton, allocates mock
inputs on CUDA, and calls ``JITFunction.warmup``.  ``warmup`` compiles but never
launches the kernel.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from .codegen import extract_python_module, validate_triton_module

__all__ = [
    "CompileCase",
    "CompileResult",
    "KernelSource",
    "RuntimeArgument",
    "TritonCompilationError",
    "TritonCompiler",
]

_CACHE_SCHEMA_VERSION = 1
_WORKER = Path(__file__).with_name("_triton_worker.py")


def _plain_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if value is not None else {}


@dataclass(frozen=True)
class KernelSource:
    """A Triton module and the ``@triton.jit`` function to compile.

    ``kind='module_file'`` reads ``path``; ``kind='code'`` uses ``code``.
    Untrusted input is AST-validated in the parent process.  ``trusted=True`` is
    intended only for source-side kernels owned by the repository and is never
    needed for a model-generated replay.
    """

    kind: Literal["module_file", "code"]
    function: str
    path: str | os.PathLike[str] | None = None
    code: str | None = None
    trusted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"module_file", "code"}:
            raise ValueError("KernelSource.kind must be 'module_file' or 'code'")
        if not self.function.isidentifier() or "__" in self.function:
            raise ValueError(f"invalid Triton function name: {self.function!r}")
        if self.kind == "module_file":
            if self.path is None or self.code is not None:
                raise ValueError("module_file source requires path and forbids code")
            object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        elif not isinstance(self.code, str) or self.path is not None:
            raise ValueError("code source requires a Python string and forbids path")

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        function: str,
        *,
        trusted: bool = False,
    ) -> "KernelSource":
        return cls("module_file", function, path=path, trusted=trusted)

    @classmethod
    def from_code(
        cls, code: str, function: str, *, trusted: bool = False
    ) -> "KernelSource":
        return cls("code", function, code=code, trusted=trusted)

    @classmethod
    def from_model_response(
        cls, response: str, function: str
    ) -> "KernelSource":
        return cls.from_code(
            extract_python_module(response, required_function=function),
            function,
            trusted=False,
        )

    @property
    def function_name(self) -> str:
        """Compatibility spelling used by some Gate1 callers."""

        return self.function

    @property
    def module_file(self) -> Path | None:
        return Path(self.path) if self.path is not None else None

    def read_code(self) -> tuple[str, str]:
        if self.kind == "code":
            assert self.code is not None
            return self.code, "<gate1-replay>"
        assert self.path is not None
        source_path = Path(self.path).resolve()
        try:
            return source_path.read_text(encoding="utf-8"), str(source_path)
        except OSError as exc:
            raise ValueError(f"cannot read Triton module {source_path}: {exc}") from exc


@dataclass(frozen=True)
class RuntimeArgument:
    """One compile-time argument used to specialize a Triton kernel."""

    kind: Literal["tensor", "scalar"]
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    value: int | float | bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"tensor", "scalar"}:
            raise ValueError("RuntimeArgument.kind must be 'tensor' or 'scalar'")
        if self.kind == "tensor":
            if self.dtype is None or self.shape is None or self.value is not None:
                raise ValueError("tensor argument requires dtype/shape and forbids value")
            shape = tuple(int(dim) for dim in self.shape)
            if any(dim < 0 for dim in shape):
                raise ValueError("tensor dimensions must be non-negative")
            object.__setattr__(self, "shape", shape)
            object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))
            if self.strides is not None:
                strides = tuple(int(stride) for stride in self.strides)
                if len(strides) != len(shape) or any(stride < 0 for stride in strides):
                    raise ValueError("tensor strides must match rank and be non-negative")
                object.__setattr__(self, "strides", strides)
        else:
            if self.value is None or self.shape is not None or self.strides is not None:
                raise ValueError("scalar argument requires value and forbids shape/strides")
            if not isinstance(self.value, (bool, int, float)):
                raise ValueError("scalar value must be bool, int, or float")
            if self.dtype is not None:
                object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))

    @classmethod
    def tensor(
        cls,
        shape: tuple[int, ...] | list[int],
        dtype: str = "float32",
        *,
        strides: tuple[int, ...] | list[int] | None = None,
    ) -> "RuntimeArgument":
        return cls(
            "tensor",
            dtype=dtype,
            shape=tuple(shape),
            strides=tuple(strides) if strides is not None else None,
        )

    @classmethod
    def scalar(
        cls, value: int | float | bool, dtype: str | None = None
    ) -> "RuntimeArgument":
        return cls("scalar", dtype=dtype, value=value)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["shape"] is not None:
            payload["shape"] = list(payload["shape"])
        if payload["strides"] is not None:
            payload["strides"] = list(payload["strides"])
        return payload


@dataclass(frozen=True)
class CompileCase:
    """Specialization, launch grid, and Layer-C values for one compilation."""

    arguments: Mapping[str, RuntimeArgument]
    grid: tuple[int, ...]
    schedule: Mapping[str, int | float | bool] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self) -> None:
        arguments = dict(self.arguments)
        if not arguments:
            raise ValueError("CompileCase.arguments must not be empty")
        for name, argument in arguments.items():
            if not isinstance(name, str) or not name.isidentifier() or "__" in name:
                raise ValueError(f"invalid argument name: {name!r}")
            if not isinstance(argument, RuntimeArgument):
                raise TypeError(f"argument {name!r} is not RuntimeArgument")
        grid = tuple(int(dim) for dim in self.grid)
        if not 1 <= len(grid) <= 3 or any(dim <= 0 for dim in grid):
            raise ValueError("grid must contain one to three positive dimensions")
        schedule = dict(self.schedule)
        for name, value in schedule.items():
            if not isinstance(name, str) or not name.isidentifier() or "__" in name:
                raise ValueError(f"invalid schedule name: {name!r}")
            if name in arguments:
                raise ValueError(f"{name!r} appears in both arguments and schedule")
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"schedule value {name!r} is not JSON scalar")
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("Gate1 Triton compiler currently supports CUDA only")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "schedule", schedule)

    def to_payload(self) -> dict[str, Any]:
        return {
            "arguments": {
                name: argument.to_payload()
                for name, argument in self.arguments.items()
            },
            "grid": list(self.grid),
            "schedule": dict(self.schedule),
            "device": self.device,
        }


@dataclass(frozen=True)
class CompileResult:
    """Compile artifact and reproducibility metadata returned by the worker."""

    ok: bool
    ttir: str | None = None
    ttgir: str | None = None
    python_interface: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: str = ""
    cache_key: str | None = None
    elapsed_ms: float = 0.0
    asm_keys: tuple[str, ...] = ()
    cache_hit: bool = False

    @property
    def versions(self) -> Mapping[str, Any]:
        """Alias emphasizing that ``compiler`` stores toolchain versions."""

        return self.compiler

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompileResult":
        return cls(
            ok=bool(payload.get("ok", False)),
            ttir=payload.get("ttir"),
            ttgir=payload.get("ttgir"),
            python_interface=_plain_mapping(payload.get("python_interface")),
            compiler=_plain_mapping(payload.get("compiler")),
            target=_plain_mapping(payload.get("target")),
            diagnostic=str(payload.get("diagnostic", "")),
            cache_key=payload.get("cache_key"),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
            asm_keys=tuple(payload.get("asm_keys", ())),
            cache_hit=bool(payload.get("cache_hit", False)),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asm_keys"] = list(self.asm_keys)
        return payload


class TritonCompilationError(RuntimeError):
    """Raised by :meth:`TritonCompiler.compile_or_raise` on frontend failure."""

    def __init__(self, result: CompileResult):
        super().__init__(result.diagnostic or "Triton compilation failed")
        self.result = result


class TritonCompiler:
    """Compile Triton kernels in a new Python process without launching them."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        timeout: float = 180.0,
        python_executable: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.timeout = float(timeout)
        self.python_executable = str(python_executable or sys.executable)
        self.extra_env = {str(key): str(value) for key, value in (env or {}).items()}

    @staticmethod
    def _installed_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _automatic_key(self, payload: Mapping[str, Any]) -> str:
        key_material = {
            "schema": _CACHE_SCHEMA_VERSION,
            "request": payload,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": self._installed_version("torch"),
            "triton": self._installed_version("triton"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        canonical = json.dumps(
            key_material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        # Hash even caller-provided keys: a cache label must never become a path.
        filename = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"
        return self.cache_dir / filename

    def _read_cache(self, key: str) -> CompileResult | None:
        if self.cache_dir is None:
            return None
        path = self._cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_schema") != _CACHE_SCHEMA_VERSION:
                return None
            result = CompileResult.from_payload(payload["result"])
            if not result.ok:
                return None
            return replace(result, cache_key=key, cache_hit=True)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, key: str, result: CompileResult) -> None:
        if self.cache_dir is None or not result.ok:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self._cache_path(key)
        payload = {
            "cache_schema": _CACHE_SCHEMA_VERSION,
            "result": replace(result, cache_hit=False).to_payload(),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.cache_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def compile(
        self,
        source: KernelSource,
        case: CompileCase,
        *,
        cache_key: str | None = None,
    ) -> CompileResult:
        """Compile one specialization and return TTIR/TTGIR plus diagnostics.

        The source is imported only by the worker.  In particular, validation
        in this process never executes the model-generated module.
        """

        if not isinstance(source, KernelSource):
            raise TypeError("source must be KernelSource")
        if not isinstance(case, CompileCase):
            raise TypeError("case must be CompileCase")
        code, filename = source.read_code()
        if not source.trusted:
            validate_triton_module(code, required_function=source.function)

        request = {
            "schema": _CACHE_SCHEMA_VERSION,
            "source": {
                "code": code,
                "filename": filename,
                "function": source.function,
                # This field participates in the evidence/cache identity even
                # though the worker does not execute policy from it.  Source
                # and replay must each compile once when their text happens to
                # be identical; a replay may never borrow a trusted-source
                # compile artifact.
                "trust_role": "trusted_source" if source.trusted else "untrusted_replay",
            },
            "case": case.to_payload(),
        }
        if cache_key is not None and (
            not isinstance(cache_key, str) or not cache_key.strip()
        ):
            raise ValueError("cache_key must be a non-empty string")
        automatic_key = self._automatic_key(request)
        # A caller-provided key is a namespace label, never authority to reuse
        # an artifact for different code/C/target/trust role.
        resolved_key = (
            automatic_key
            if cache_key is None
            else hashlib.sha256(
                (cache_key + "\x00" + automatic_key).encode("utf-8")
            ).hexdigest()
        )
        cached = self._read_cache(resolved_key)
        if cached is not None:
            return cached

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="intentir-gate1-") as temporary:
            work = Path(temporary)
            request_path = work / "request.json"
            result_path = work / "result.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True), encoding="utf-8"
            )

            environment = dict(os.environ)
            environment.update(self.extra_env)
            # These flags are part of the Gate1 evidence contract.  Always
            # compile and retain the textual IR; the worker also reads asm
            # directly so Triton's filesystem dump is supporting evidence only.
            environment.update(
                {
                    "TRITON_KERNEL_DUMP": "1",
                    "TRITON_ALWAYS_COMPILE": "1",
                    "TRITON_STORE_BINARY_ONLY": "0",
                    "TRITON_CACHE_DIR": str(work / "triton-cache"),
                    "TRITON_DUMP_DIR": str(work / "triton-dump"),
                    "PYTHONHASHSEED": "0",
                }
            )
            command = [
                self.python_executable,
                str(_WORKER),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ]
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = (time.monotonic() - started) * 1000.0
                diagnostic = f"Triton compile worker timed out after {self.timeout:.1f}s"
                if exc.stderr:
                    diagnostic += "\n" + str(exc.stderr)
                return CompileResult(
                    ok=False,
                    diagnostic=diagnostic,
                    cache_key=resolved_key,
                    elapsed_ms=elapsed,
                )

            elapsed = (time.monotonic() - started) * 1000.0
            try:
                worker_payload = json.loads(result_path.read_text(encoding="utf-8"))
                result = CompileResult.from_payload(worker_payload)
            except (OSError, ValueError, TypeError) as exc:
                diagnostic = (
                    "Triton compile worker did not produce a valid result: "
                    f"{exc}"
                )
                if process.stdout:
                    diagnostic += "\nworker stdout:\n" + process.stdout
                if process.stderr:
                    diagnostic += "\nworker stderr:\n" + process.stderr
                result = CompileResult(ok=False, diagnostic=diagnostic)

            if process.returncode != 0:
                diagnostic = result.diagnostic or (
                    f"Triton compile worker exited with status {process.returncode}"
                )
                if process.stdout:
                    diagnostic += "\nworker stdout:\n" + process.stdout
                if process.stderr:
                    diagnostic += "\nworker stderr:\n" + process.stderr
                result = replace(result, ok=False, diagnostic=diagnostic)

        result = replace(
            result,
            cache_key=resolved_key,
            elapsed_ms=result.elapsed_ms or elapsed,
            cache_hit=False,
        )
        self._write_cache(resolved_key, result)
        return result

    def compile_or_raise(
        self,
        source: KernelSource,
        case: CompileCase,
        *,
        cache_key: str | None = None,
    ) -> CompileResult:
        result = self.compile(source, case, cache_key=cache_key)
        if not result.ok:
            raise TritonCompilationError(result)
        return result
