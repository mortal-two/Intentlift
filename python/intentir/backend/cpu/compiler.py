"""Content-addressed compiler for the Gate2 scalar CPU reference module."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..toolchain.tools import (
    CLANG,
    INTENT_OPT,
    LLVM_LIB,
    MLIR_OPT,
    MLIR_TRANSLATE,
)
from .abi import CPU_MEMREF_ABI_PROTOCOL


CPU_COMPILER_PROTOCOL = "intentir_cpu_mlir_shared_object_v1"
CPU_EXECUTION_EVIDENCE_SCHEMA = "intentir_cpu_execution_evidence_v1"

CPU_LOWERING_PIPELINE = (
    "-one-shot-bufferize=bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map",
    "-buffer-deallocation-pipeline",
    "-convert-bufferization-to-memref",
    "-convert-linalg-to-loops",
    "-convert-math-to-libm",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-expand-strided-metadata",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
)

_CLANG_FLAGS = (
    "-O2",
    "-fPIC",
    "-shared",
    "-x",
    "ir",
    "-",
    "-lm",
    "-Wl,-z,defs",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _c_runner_utils() -> Path:
    try:
        return (LLVM_LIB / "libmlir_c_runner_utils.so").resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "required MLIR C runner utility library is unavailable: "
            f"{LLVM_LIB / 'libmlir_c_runner_utils.so'}"
        ) from exc


def _canonical_digest(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _resolve_tool(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        found = shutil.which(value)
        if found is None:
            raise RuntimeError(f"required CPU lowering tool is unavailable: {value}")
        candidate = Path(found)
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"required CPU lowering tool is unavailable: {candidate}"
        ) from exc


@lru_cache(maxsize=None)
def _tool_identity(value: str) -> dict[str, Any]:
    path = _resolve_tool(value)
    stat = path.stat()
    version = subprocess.run(
        [str(path), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode:
        raise RuntimeError(
            f"could not identify CPU lowering tool {path}: "
            f"{version.stderr.strip()[:500]}"
        )
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
        "version": version.stdout.strip(),
    }


def _toolchain_identity() -> dict[str, Any]:
    return {
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "pointer_bits": struct.calcsize("P") * 8,
        },
        "intent_opt": _tool_identity(str(INTENT_OPT)),
        "mlir_opt": _tool_identity(str(MLIR_OPT)),
        "mlir_translate": _tool_identity(str(MLIR_TRANSLATE)),
        "clang": _tool_identity(str(CLANG)),
        "mlir_c_runner_utils": _file_identity(_c_runner_utils()),
    }


def _cache_root() -> Path:
    configured = os.environ.get("INTENTIR_CPU_CACHE_DIR")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir())
        / f"intentir-cpu-cache-v1-{os.getuid()}"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _run_text(
    argv: Sequence[str],
    source: str,
    *,
    stage: str,
) -> str:
    completed = subprocess.run(
        list(argv),
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{stage} failed: {diagnostic[:2000]}")
    return completed.stdout


@dataclass(frozen=True)
class CompiledCPUModule:
    """One immutable cached scalar CPU shared object and its provenance."""

    shared_object: Path
    source_digest: str
    lowering_pipeline_digest: str
    toolchain_digest: str
    cache_key: str
    shared_object_digest: str
    cache_hit: bool

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "schema": CPU_EXECUTION_EVIDENCE_SCHEMA,
            "compiler_protocol": CPU_COMPILER_PROTOCOL,
            "abi_protocol": CPU_MEMREF_ABI_PROTOCOL,
            "scalar_module_digest": self.source_digest,
            "lowering_pipeline_digest": self.lowering_pipeline_digest,
            "toolchain_digest": self.toolchain_digest,
            "cache_key": self.cache_key,
            "shared_object_digest": self.shared_object_digest,
            # Telemetry is deliberately not part of any stable digest above.
            "cache_hit": self.cache_hit,
        }


def compile_cpu_module(source: str) -> CompiledCPUModule:
    """Lower one static MLIR module and return a cached shared object.

    The key contains module text, compiler pipeline, flags, ABI protocol and
    toolchain identity.  Runtime input values and random seeds never enter it.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("CPU compilation requires non-empty MLIR source")

    source_digest = _sha256_text(source)
    c_runner_utils = _c_runner_utils()
    runtime_link_flags = (
        # ``-x ir`` in the fixed stdin flags otherwise also classifies the
        # following shared library as textual LLVM IR.
        "-x",
        "none",
        str(c_runner_utils),
        f"-Wl,-rpath,{c_runner_utils.parent}",
    )
    pipeline_identity = {
        "compiler_protocol": CPU_COMPILER_PROTOCOL,
        "abi_protocol": CPU_MEMREF_ABI_PROTOCOL,
        "intent_opt_passes": [
            "-convert-intent-to-linalg",
            "-expand-rsqrt-for-libm",
        ],
        "mlir_pipeline": list(CPU_LOWERING_PIPELINE),
        "clang_flags": list(_CLANG_FLAGS),
        "runtime_link_flags": list(runtime_link_flags),
    }
    lowering_pipeline_digest = _canonical_digest(pipeline_identity)
    tools = _toolchain_identity()
    toolchain_digest = _canonical_digest(tools)
    cache_key = _canonical_digest(
        {
            "source_digest": source_digest,
            "lowering_pipeline_digest": lowering_pipeline_digest,
            "toolchain_digest": toolchain_digest,
        }
    )

    root = _cache_root()
    shared_object = root / f"{cache_key}.so"
    manifest_path = root / f"{cache_key}.json"
    lock_path = root / f"{cache_key}.lock"
    stable_manifest = {
        "schema": CPU_EXECUTION_EVIDENCE_SCHEMA,
        "compiler_protocol": CPU_COMPILER_PROTOCOL,
        "abi_protocol": CPU_MEMREF_ABI_PROTOCOL,
        "source_digest": source_digest,
        "lowering_pipeline_digest": lowering_pipeline_digest,
        "toolchain_digest": toolchain_digest,
        "cache_key": cache_key,
        "shared_object_name": shared_object.name,
    }

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cache_hit = False
        shared_object_digest = ""
        if shared_object.is_file() and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                shared_object_digest = _sha256_file(shared_object)
                cache_hit = (
                    all(
                        manifest.get(key) == value
                        for key, value in stable_manifest.items()
                    )
                    and manifest.get("shared_object_digest")
                    == shared_object_digest
                )
            except (AttributeError, OSError, json.JSONDecodeError):
                cache_hit = False
        if not cache_hit:
            intent_mlir = _run_text(
                [
                    str(_resolve_tool(str(INTENT_OPT))),
                    "-convert-intent-to-linalg",
                    "-expand-rsqrt-for-libm",
                ],
                source,
                stage="intent-opt",
            )
            llvm_mlir = _run_text(
                [
                    str(_resolve_tool(str(MLIR_OPT))),
                    *CPU_LOWERING_PIPELINE,
                ],
                intent_mlir,
                stage="mlir-opt",
            )
            llvm_ir = _run_text(
                [str(_resolve_tool(str(MLIR_TRANSLATE))), "--mlir-to-llvmir"],
                llvm_mlir,
                stage="mlir-translate",
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_key}.", suffix=".so", dir=root
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                completed = subprocess.run(
                    [
                        str(_resolve_tool(str(CLANG))),
                        *_CLANG_FLAGS,
                        *runtime_link_flags,
                        "-o",
                        str(temporary),
                    ],
                    input=llvm_ir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode:
                    diagnostic = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(f"clang failed: {diagnostic[:2000]}")
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("clang produced no CPU shared object")
                shared_object_digest = _sha256_file(temporary)
                os.replace(temporary, shared_object)
                manifest = {
                    **stable_manifest,
                    "shared_object_digest": shared_object_digest,
                }
                descriptor, manifest_temporary_name = tempfile.mkstemp(
                    prefix=f".{cache_key}.", suffix=".json", dir=root
                )
                manifest_temporary = Path(manifest_temporary_name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(
                            manifest,
                            stream,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(manifest_temporary, manifest_path)
                finally:
                    if manifest_temporary.exists():
                        manifest_temporary.unlink()
            finally:
                if temporary.exists():
                    temporary.unlink()

    return CompiledCPUModule(
        shared_object=shared_object,
        source_digest=source_digest,
        lowering_pipeline_digest=lowering_pipeline_digest,
        toolchain_digest=toolchain_digest,
        cache_key=cache_key,
        shared_object_digest=shared_object_digest,
        cache_hit=cache_hit,
    )


__all__ = [
    "CPU_COMPILER_PROTOCOL",
    "CPU_EXECUTION_EVIDENCE_SCHEMA",
    "CPU_LOWERING_PIPELINE",
    "CompiledCPUModule",
    "compile_cpu_module",
]
