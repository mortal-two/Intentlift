#!/usr/bin/env python3
"""Measure the historical multi-frontend onboarding commits by ownership.

The checkpoints added TileLang and CUDA together, so this tool reports exact
language-specific and shared deltas instead of inventing a per-frontend split.
Tests and documentation remain visible but are excluded from the production
headline.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    "d30753e7a1c7aafbd6d28632bf9e46e119249efc",
    "eb58f3aed66b5f02b6a008e072ca9fecd3d2f367",
    "a21f90f27bffe9a6b79c17c440aa6d74ebe5b701",
)

_TILELANG = frozenset(
    {
        "python/intentir/verify/gate1/_tilelang_worker.py",
        "python/intentir/verify/gate1/tilelang.py",
        "python/intentir/verify/gate1/tilelang_codegen.py",
        "python/intentir/verify/gate1/tilelang_compiler.py",
    }
)
_CUDA = frozenset(
    {
        "python/intentir/verify/gate1/cuda_codegen.py",
        "python/intentir/verify/gate1/cuda_compiler.py",
        "python/intentir/verify/gate1/ptx.py",
    }
)
_MULTIFRONTEND_SHARED = frozenset(
    {
        "python/intentir/verify/frontend_dataset.py",
        "python/intentir/verify/frontend_workloads.py",
        "python/intentir/verify/gate1/frontend.py",
        "python/intentir/verify/gate1/native_orchestrate.py",
        "python/intentir/verify/gate2/native.py",
        "python/intentir/verify/workload_registry.py",
        "tools/frontend_dataset.py",
        "tools/scrc_frontend.py",
    }
)
_SCRC_SHARED = frozenset(
    {
        "python/intentir/extraction/lifters/miniswe.py",
        "python/intentir/extraction/lifters/scrc.py",
        "python/intentir/verify/adapters.py",
        "python/intentir/verify/gate1/profile.py",
        "python/intentir/verify/gate1/prompt.py",
        "python/intentir/verify/gate1/replay.py",
        "python/intentir/verify/scrc_workload_run.py",
        "tools/scrc_gate1_recheck.py",
    }
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-c", "core.quotepath=false", *arguments),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _category(path: str) -> str:
    if path.startswith("python/tests/"):
        return "historical_tests"
    if path.startswith(("doc/", ".trellis/")):
        return "docs_specs"
    if path in _TILELANG:
        return "tilelang_specific"
    if path in _CUDA:
        return "cuda_specific"
    if path in _MULTIFRONTEND_SHARED:
        return "multifrontend_shared"
    if path in _SCRC_SHARED:
        return "scrc_shared_core"
    if path.startswith(
        ("python/intentir/backend/", "include/", "lib/", "test/")
    ):
        return "backend"
    raise ValueError(f"unclassified onboarding path: {path}")


def _measure(commit: str) -> dict[str, Any]:
    identity = _git("show", "-s", "--format=%H%x00%s", commit).rstrip("\n")
    commit_hash, subject = identity.split("\x00", 1)
    categories: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    for line in _git(
        "show", "--format=", "--numstat", "--no-renames", commit
    ).splitlines():
        if not line:
            continue
        added, deleted, path = line.split("\t", 2)
        if added == "-" or deleted == "-":
            raise ValueError(f"binary onboarding delta is not LoC-countable: {path}")
        category = _category(path)
        row = {
            "path": path,
            "category": category,
            "additions": int(added),
            "deletions": int(deleted),
        }
        files.append(row)
        aggregate = categories.setdefault(
            category, {"additions": 0, "deletions": 0, "files": 0}
        )
        aggregate["additions"] += row["additions"]
        aggregate["deletions"] += row["deletions"]
        aggregate["files"] += 1
    return {
        "commit": commit_hash,
        "subject": subject,
        "categories": dict(sorted(categories.items())),
        "files": files,
    }


def main() -> int:
    commits = [_measure(commit) for commit in CHECKPOINTS]
    aggregate: dict[str, dict[str, int]] = {}
    unique_files: dict[str, set[str]] = {}
    for commit in commits:
        for row in commit["files"]:
            unique_files.setdefault(row["category"], set()).add(row["path"])
        for category, counts in commit["categories"].items():
            target = aggregate.setdefault(
                category,
                {"additions": 0, "deletions": 0, "file_touches": 0},
            )
            target["additions"] += int(counts["additions"])
            target["deletions"] += int(counts["deletions"])
            target["file_touches"] += int(counts["files"])
    for category, paths in unique_files.items():
        aggregate[category]["unique_files"] = len(paths)
    production_categories = (
        "tilelang_specific",
        "cuda_specific",
        "multifrontend_shared",
        "scrc_shared_core",
        "backend",
    )
    production = {
        field: 0
        for field in ("additions", "deletions", "file_touches")
    }
    production_paths: set[str] = set()
    for category in production_categories:
        for field in production:
            production[field] += aggregate.get(category, {}).get(field, 0)
        production_paths.update(unique_files.get(category, set()))
    production["unique_files"] = len(production_paths)
    payload = {
        "protocol": "intentir.paper.multifrontend-onboarding-loc.v1",
        "checkpoints": commits,
        "aggregate": dict(sorted(aggregate.items())),
        "production_headline": production,
        "production_categories": list(production_categories),
        "backend_delta": aggregate.get(
            "backend",
            {
                "additions": 0,
                "deletions": 0,
                "file_touches": 0,
                "unique_files": 0,
            },
        ),
        "caveats": [
            "TileLang and CUDA were historically onboarded together; shared LoC is not assigned to either frontend.",
            "Counts are commit-level physical LoC, not person-hours.",
            "Historical tests and docs/specs are reported but excluded from the production headline.",
            "The checkpoints are measured individually; intervening ancestry is not claimed as frontend-only work.",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
