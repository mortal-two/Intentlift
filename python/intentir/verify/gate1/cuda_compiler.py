"""Compile-only CUDA C++ frontend for SCRC Gate1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Literal, Mapping

from .cuda_codegen import validate_cuda_module


@dataclass(frozen=True)
class FrontendParameter:
    kind: Literal["pointer", "scalar", "opaque"]
    dtype: str
    rank: int = 0
    mode: Literal["input", "output", "inout", "value"] = "value"

    def __post_init__(self) -> None:
        if self.kind not in {"pointer", "scalar", "opaque"}:
            raise ValueError("unsupported frontend parameter kind")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("frontend parameter dtype must be non-empty")
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("frontend parameter rank must be non-negative")
        if self.kind == "pointer" and self.mode not in {"input", "output", "inout"}:
            raise ValueError("pointer mode must describe an effect")
        if self.kind != "pointer" and self.mode != "value":
            raise ValueError("non-pointer parameters must use value mode")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CudaCompileCase:
    parameters: tuple[FrontendParameter, ...]
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    arch: str = "sm_80"
    compiler_options: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parameters = tuple(
            item if isinstance(item, FrontendParameter) else FrontendParameter(**item)
            for item in self.parameters
        )
        if not parameters:
            raise ValueError("CUDA compile case requires at least one parameter")
        for label, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"CUDA {label} must contain three positive dimensions")
            object.__setattr__(self, label, dims)
        schedule = dict(self.schedule)
        if any(
            not isinstance(key, str)
            or not key.isidentifier()
            or "__" in key
            for key in schedule
        ):
            raise ValueError("CUDA schedule keys must be safe identifiers")
        if any(
            not isinstance(value, (bool, int, float, str))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in schedule.values()
        ):
            raise TypeError("CUDA schedule values must be finite JSON scalars")
        allowed = (
            "--fmad=",
            "--ftz=",
            "--maxrregcount=",
            "--prec-div=",
            "--prec-sqrt=",
            "--use_fast_math",
        )
        options = tuple(str(item) for item in self.compiler_options)
        if any(not item.startswith(allowed) for item in options):
            raise ValueError("CUDA compiler option is outside the Gate1 profile")
        arch = _normalize_arch(self.arch)
        include_dirs = tuple(
            str(Path(item).expanduser().resolve()) for item in self.include_dirs
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "compiler_options", options)
        object.__setattr__(self, "include_dirs", include_dirs)
        object.__setattr__(self, "arch", arch)

    @property
    def interface(self) -> dict[str, Any]:
        return {
            "parameters": [item.to_json() for item in self.parameters],
            "grid_rank": sum(int(item != 1) for item in self.grid) or 1,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameters": [item.to_json() for item in self.parameters],
            "grid": list(self.grid),
            "block": list(self.block),
            "schedule": dict(self.schedule),
            "arch": self.arch,
            "compiler_options": list(self.compiler_options),
            "include_dirs": list(self.include_dirs),
        }


@dataclass(frozen=True)
class CudaKernelSource:
    kind: Literal["module_file", "code"]
    function: str
    path: str | os.PathLike[str] | None = None
    code: str | None = None
    trusted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"module_file", "code"}:
            raise ValueError("unsupported CUDA kernel source kind")
        if not self.function.isidentifier() or "__" in self.function:
            raise ValueError("invalid CUDA kernel function name")
        if self.kind == "module_file":
            if self.path is None or self.code is not None:
                raise ValueError("CUDA module_file requires only path")
            object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        elif not isinstance(self.code, str) or self.path is not None:
            raise ValueError("CUDA code source requires only code")
        if not self.trusted:
            assert self.code is not None
            validate_cuda_module(self.code, required_kernel=self.function)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], function: str) -> "CudaKernelSource":
        return cls("module_file", function, path=path, trusted=True)

    @classmethod
    def from_code(
        cls,
        code: str,
        function: str,
        *,
        trusted: bool = False,
    ) -> "CudaKernelSource":
        return cls("code", function, code=code, trusted=trusted)

    @classmethod
    def from_replay(cls, code: str, function: str) -> "CudaKernelSource":
        return cls.from_code(code, function, trusted=False)

    def read_code(self) -> str:
        if self.code is not None:
            return self.code
        assert self.path is not None
        try:
            return Path(self.path).resolve().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read CUDA source {self.path}: {exc}") from exc


@dataclass(frozen=True)
class CudaCompileResult:
    ok: bool
    ptx: str | None = None
    entry: str | None = None
    interface: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: str = ""
    cache_key: str | None = None
    elapsed_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ptx": self.ptx,
            "entry": self.entry,
            "interface": dict(self.interface),
            "compiler": dict(self.compiler),
            "target": dict(self.target),
            "diagnostic": self.diagnostic,
            "cache_key": self.cache_key,
            "elapsed_ms": self.elapsed_ms,
        }


def _normalize_arch(raw: str) -> str:
    value = str(raw or "").strip().lower()
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        raise ValueError("CUDA arch must contain a compute capability")
    return f"sm_{digits}"


def _schedule_define(name: str, value: int | float | bool | str) -> str:
    if isinstance(value, bool):
        literal = "1" if value else "0"
    elif isinstance(value, (int, float)):
        literal = repr(value)
    else:
        literal = json.dumps(value, ensure_ascii=True)
    return f"-D{name}={literal}"


def find_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    candidates: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(key)
        if home:
            candidates.append(Path(home) / "bin" / "nvcc")
    candidates.extend((Path("/usr/local/cuda/bin/nvcc"), Path("/opt/cuda/bin/nvcc")))
    for pattern in ("/usr/local/cuda-*/bin/nvcc", "/opt/cuda-*/bin/nvcc"):
        candidates.extend(Path(item) for item in sorted(glob.glob(pattern), reverse=True))
    return next((str(item) for item in candidates if item.is_file()), None)


class CudaCompiler:
    """Compile a trusted Source or validated Replay to PTX without launching it."""

    def __init__(self, *, nvcc: str | None = None, timeout: float = 180.0) -> None:
        if timeout <= 0:
            raise ValueError("CUDA compiler timeout must be positive")
        self.nvcc = str(nvcc) if nvcc is not None else find_nvcc()
        self.timeout = float(timeout)

    def _version(self) -> str:
        if self.nvcc is None:
            return "unavailable"
        try:
            result = subprocess.run(
                [self.nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            lines = (result.stdout or result.stderr).strip().splitlines()
            return lines[-1] if lines else "nvcc"
        except Exception:
            return "nvcc"

    def compile(self, source: CudaKernelSource, case: CudaCompileCase) -> CudaCompileResult:
        started = time.perf_counter()
        if self.nvcc is None:
            return CudaCompileResult(
                False,
                diagnostic="nvcc is unavailable",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        try:
            code = source.read_code()
        except Exception as exc:
            return CudaCompileResult(
                False,
                diagnostic=f"source read failed: {type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        key_payload = {
            "source": code,
            "entry": source.function,
            "case": case.to_payload(),
            "nvcc": self._version(),
        }
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory(prefix="intentir-cuda-gate1-") as raw_dir:
            directory = Path(raw_dir)
            source_path = directory / "kernel.cu"
            ptx_path = directory / "kernel.ptx"
            source_path.write_text(code, encoding="utf-8")
            command = [
                self.nvcc,
                "--ptx",
                "--std=c++17",
                "-O3",
                "-lineinfo",
                f"-arch={case.arch}",
                *(
                    _schedule_define(name, value)
                    for name, value in sorted(case.schedule.items())
                ),
                *case.compiler_options,
                *(
                    item
                    for directory in case.include_dirs
                    for item in ("-I", directory)
                ),
                str(source_path),
                "-o",
                str(ptx_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    cwd=directory,
                )
            except subprocess.TimeoutExpired:
                return CudaCompileResult(
                    False,
                    entry=source.function,
                    diagnostic=f"CUDA compilation timed out after {self.timeout:g}s",
                    cache_key=cache_key,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            if result.returncode != 0 or not ptx_path.is_file():
                diagnostic = (result.stderr or result.stdout or "nvcc failed").strip()
                return CudaCompileResult(
                    False,
                    entry=source.function,
                    compiler={"name": "nvcc", "version": self._version()},
                    target={"arch": case.arch},
                    diagnostic=diagnostic,
                    cache_key=cache_key,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            return CudaCompileResult(
                True,
                ptx=ptx_path.read_text(encoding="utf-8"),
                entry=source.function,
                interface=case.interface,
                compiler={"name": "nvcc", "version": self._version()},
                target={"arch": case.arch},
                cache_key=cache_key,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )


__all__ = [
    "CudaCompileCase",
    "CudaCompileResult",
    "CudaCompiler",
    "CudaKernelSource",
    "FrontendParameter",
    "find_nvcc",
]
