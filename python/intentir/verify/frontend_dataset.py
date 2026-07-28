"""External Dataset catalog for the active TileLang and CUDA frontend corpus."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

MULTIFRONTEND_DATASET_PROTOCOL = "intentir.multifrontend-dataset.v1"
_CUDA_KERNEL_RE = re.compile(
    r"(?:extern\s+\"C\"\s+)?__global__\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+"
    r"(?:__launch_bounds__\([^)]*\)\s+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)


@dataclass(frozen=True)
class FrontendDatasetKernel:
    case_id: str
    frontend: str
    path: Path
    symbol: str
    source_kind: str
    location: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "frontend": self.frontend,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "location": self.location,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FrontendDatasetValidation:
    kernels: tuple[FrontendDatasetKernel, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.kernels:
            result[item.frontend] = result.get(item.frontend, 0) + 1
        return dict(sorted(result.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": MULTIFRONTEND_DATASET_PROTOCOL,
            "passed": self.passed,
            "counts": self.counts,
            "errors": list(self.errors),
            "kernels": [item.to_json() for item in self.kernels],
        }


@dataclass(frozen=True)
class FrontendDatasetCatalog:
    root: Path
    kernels: tuple[FrontendDatasetKernel, ...]

    @property
    def by_id(self) -> dict[str, FrontendDatasetKernel]:
        return {item.case_id: item for item in self.kernels}

    def select(self, *, frontend: str | None = None) -> tuple[FrontendDatasetKernel, ...]:
        if frontend is None:
            return self.kernels
        if frontend not in {"tilelang", "cuda"}:
            raise ValueError(f"unsupported Dataset frontend {frontend!r}")
        return tuple(item for item in self.kernels if item.frontend == frontend)

    def validate(self) -> FrontendDatasetValidation:
        errors: list[str] = []
        ids = [item.case_id for item in self.kernels]
        if len(ids) != len(set(ids)):
            errors.append("multi-frontend Dataset case ids must be unique")
        counts = {frontend: len(self.select(frontend=frontend)) for frontend in ("tilelang", "cuda")}
        if counts["tilelang"] < 30:
            errors.append("multi-frontend Dataset must retain at least 30 TileLang kernels")
        if counts["cuda"] < 30:
            errors.append("multi-frontend Dataset must retain at least 30 CUDA kernels")
        manifest_path = self.root / "manifest.json"
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded_manifest if isinstance(loaded_manifest, dict) else {}
        except (OSError, ValueError, UnicodeError):
            manifest = {}
            errors.append("multi-frontend Dataset manifest is missing or invalid")
        if manifest.get("protocol") != MULTIFRONTEND_DATASET_PROTOCOL:
            errors.append("multi-frontend Dataset manifest protocol differs")
        if (manifest.get("tilelang") or {}).get("version") != "0.1.8":
            errors.append("multi-frontend Dataset must pin TileLang 0.1.8")
        for item in self.kernels:
            try:
                item.path.resolve().relative_to(self.root.resolve())
            except ValueError:
                errors.append(f"{item.case_id} escapes the external Dataset root")
            if not item.path.is_file():
                errors.append(f"{item.case_id} source is missing")
            if (
                item.frontend == "cuda"
                and item.metadata.get("origin") == "tilelang_export"
            ):
                metadata_path = item.path.with_suffix(".json")
                ptx_path = item.path.with_suffix(".ptx")
                if not metadata_path.is_file():
                    errors.append(f"{item.case_id} snapshot metadata is missing")
                if not ptx_path.is_file():
                    errors.append(f"{item.case_id} PTX snapshot is missing")
                if item.metadata.get("snapshot_metadata_error"):
                    errors.append(f"{item.case_id} snapshot metadata is unreadable")
                if item.metadata.get("entry_name") != item.symbol:
                    errors.append(f"{item.case_id} snapshot entry differs from CUDA source")
                if item.metadata.get("protocol") != "intentir.tilelang-cuda-snapshot.v2":
                    errors.append(f"{item.case_id} snapshot protocol is not v2")
                compiler = item.metadata.get("compiler")
                if not isinstance(compiler, dict) or compiler.get("version") != "0.1.8":
                    errors.append(f"{item.case_id} snapshot compiler is not TileLang 0.1.8")
                cuda_abi = item.metadata.get("cuda_abi")
                if not isinstance(cuda_abi, dict) or not cuda_abi.get("parameters"):
                    errors.append(f"{item.case_id} snapshot CUDA ABI is missing")
                if item.metadata.get("include_roots") != [
                    "tilelang/src",
                    "tilelang/3rdparty/cutlass/include",
                ]:
                    errors.append(f"{item.case_id} snapshot include roots are incomplete")
                if ptx_path.is_file():
                    expected = item.metadata.get("ptx_sha256")
                    actual = hashlib.sha256(ptx_path.read_bytes()).hexdigest()
                    if expected != actual:
                        errors.append(f"{item.case_id} PTX snapshot digest differs")
                expected = item.metadata.get("cuda_sha256")
                actual = hashlib.sha256(item.path.read_bytes()).hexdigest()
                if expected != actual:
                    errors.append(f"{item.case_id} CUDA snapshot digest differs")
        return FrontendDatasetValidation(self.kernels, tuple(errors))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": MULTIFRONTEND_DATASET_PROTOCOL,
            "root": str(self.root),
            "kernels": [item.to_json() for item in self.kernels],
        }


def _tilelang_kernels(root: Path) -> list[FrontendDatasetKernel]:
    source_root = root / "tilelang" / "ops"
    rows: list[FrontendDatasetKernel] = []
    for path in sorted(source_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        factories = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("make_")
            and (node.name.endswith("_prim_func") or "kernel_prim_func" in node.name)
        ]
        if len(factories) != 1:
            continue
        rows.append(
            FrontendDatasetKernel(
                f"tilelang:{path.stem}",
                "tilelang",
                path.resolve(),
                factories[0],
                "factory_file",
                f"dataset://TianchenIR-MultiFrontend/tilelang/ops/{path.name}",
                {},
            )
        )
    return rows


def _cuda_kernels(root: Path) -> list[FrontendDatasetKernel]:
    rows: list[FrontendDatasetKernel] = []
    sources = [
        ("handwritten", path)
        for path in sorted((root / "cuda" / "ops").glob("*.cu"))
    ]
    sources.extend(
        ("tilelang_export", path)
        for path in sorted((root / "cuda" / "generated").glob("*.cu"))
    )
    for origin, path in sources:
        try:
            code = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        names = list(dict.fromkeys(_CUDA_KERNEL_RE.findall(code)))
        if len(names) != 1:
            continue
        metadata: dict[str, Any] = {"origin": origin}
        if origin == "tilelang_export":
            meta_path = path.with_suffix(".json")
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata.update(loaded)
            except (OSError, ValueError, UnicodeError):
                metadata["snapshot_metadata_error"] = str(meta_path)
        case_name = path.stem if origin == "handwritten" else f"tilelang_{path.stem}"
        relative = path.relative_to(root)
        rows.append(
            FrontendDatasetKernel(
                f"cuda:{case_name}",
                "cuda",
                path.resolve(),
                names[0],
                "module_file",
                f"dataset://TianchenIR-MultiFrontend/{relative.as_posix()}",
                metadata,
            )
        )
    return rows


def load_frontend_dataset(
    dataset_root: str | os.PathLike[str] | None = None,
) -> FrontendDatasetCatalog:
    base = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else default_frontend_dataset_root()
    )
    root = (
        base
        if base.name == "TianchenIR-MultiFrontend"
        or (
            (base / "manifest.json").is_file()
            and (base / "tilelang").is_dir()
            and (base / "cuda").is_dir()
        )
        else base / "TianchenIR-MultiFrontend"
    )
    kernels = tuple([*_tilelang_kernels(root), *_cuda_kernels(root)])
    return FrontendDatasetCatalog(root, kernels)


def resolve_cuda_include_dirs(kernel: FrontendDatasetKernel) -> tuple[str, ...]:
    """Resolve portable Dataset include roots against the active installation."""
    if kernel.frontend != "cuda":
        raise ValueError("include roots are only defined for CUDA Dataset cases")
    roots = tuple(str(item) for item in kernel.metadata.get("include_roots") or ())
    if not roots:
        return tuple(
            str(Path(item).expanduser().resolve())
            for item in kernel.metadata.get("include_dirs") or ()
        )
    spec = importlib.util.find_spec("tilelang")
    if spec is None or spec.origin is None:
        raise RuntimeError("TileLang is required to resolve generated CUDA headers")
    package = Path(spec.origin).resolve().parent
    resolved: list[str] = []
    for root in roots:
        prefix = "tilelang/"
        if not root.startswith(prefix):
            raise ValueError(f"unsupported Dataset include root {root!r}")
        path = (package / root.removeprefix(prefix)).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset include root is unavailable: {path}")
        resolved.append(str(path))
    return tuple(resolved)


def default_frontend_dataset_root() -> Path:
    configured = os.environ.get("INTENTIR_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    return repository.parent / "Dataset"


__all__ = [
    "FrontendDatasetCatalog",
    "FrontendDatasetKernel",
    "FrontendDatasetValidation",
    "MULTIFRONTEND_DATASET_PROTOCOL",
    "default_frontend_dataset_root",
    "load_frontend_dataset",
    "resolve_cuda_include_dirs",
]
