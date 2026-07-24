"""External Dataset catalog for the active TileLang and CUDA frontend corpus."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

MULTIFRONTEND_DATASET_PROTOCOL = "intentir.multifrontend-dataset.v1"
_CUDA_KERNEL_RE = re.compile(
    r"(?:extern\s+\"C\"\s+)?__global__\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+"
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

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "frontend": self.frontend,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "location": self.location,
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
        if counts["cuda"] < 4:
            errors.append("multi-frontend Dataset must retain at least four CUDA kernels")
        for item in self.kernels:
            try:
                item.path.resolve().relative_to(self.root.resolve())
            except ValueError:
                errors.append(f"{item.case_id} escapes the external Dataset root")
            if not item.path.is_file():
                errors.append(f"{item.case_id} source is missing")
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
            )
        )
    return rows


def _cuda_kernels(root: Path) -> list[FrontendDatasetKernel]:
    source_root = root / "cuda" / "ops"
    rows: list[FrontendDatasetKernel] = []
    for path in sorted(source_root.glob("*.cu")):
        try:
            code = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        names = _CUDA_KERNEL_RE.findall(code)
        if len(names) != 1:
            continue
        rows.append(
            FrontendDatasetKernel(
                f"cuda:{path.stem}",
                "cuda",
                path.resolve(),
                names[0],
                "module_file",
                f"dataset://TianchenIR-MultiFrontend/cuda/ops/{path.name}",
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
    root = base / "TianchenIR-MultiFrontend"
    kernels = tuple([*_tilelang_kernels(root), *_cuda_kernels(root)])
    return FrontendDatasetCatalog(root, kernels)


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
]
