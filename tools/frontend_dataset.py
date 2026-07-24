#!/usr/bin/env python3
"""Inspect and validate the external CUDA/TileLang source Dataset catalog."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from intentir.verify.frontend_dataset import load_frontend_dataset
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompiler,
    CudaKernelSource,
    FrontendParameter,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompiler,
    TileLangKernelSource,
)


def _probe_tilelang(catalog, *, workers: int, timeout: float) -> dict:
    rows = catalog.select(frontend="tilelang")
    # Factory compilation is shape/schedule self-contained.  This placeholder
    # parameter only satisfies the compile-case type; the worker reports the
    # actual PrimFunc/PTX interface in its result.
    case = TileLangCompileCase((FrontendParameter("pointer", "f32", 1, "input"),))

    def probe(row):
        result = TileLangCompiler(timeout=timeout).compile(
            TileLangKernelSource.from_factory_file(row.path, row.symbol),
            case,
        )
        lines = (result.diagnostic or "").splitlines()
        return {
            **row.to_json(),
            "compile_ok": result.ok,
            "diagnostic": lines[0] if lines else "",
            "compiler": dict(result.compiler),
            "target": dict(result.target),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    passed = sum(bool(item["compile_ok"]) for item in results)
    return {
        "protocol": "intentir.tilelang-dataset-compile-probe.v1",
        "dataset_root": str(catalog.root),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


def _probe_cuda(catalog, *, workers: int, timeout: float, arch: str) -> dict:
    rows = catalog.select(frontend="cuda")
    case = CudaCompileCase(
        (FrontendParameter("pointer", "f32", 1, "input"),),
        arch=arch,
    )

    def probe(row):
        result = CudaCompiler(timeout=timeout).compile(
            CudaKernelSource.from_file(row.path, row.symbol),
            case,
        )
        lines = (result.diagnostic or "").splitlines()
        return {
            **row.to_json(),
            "compile_ok": result.ok,
            "diagnostic": lines[0] if lines else "",
            "compiler": dict(result.compiler),
            "target": dict(result.target),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    passed = sum(bool(item["compile_ok"]) for item in results)
    return {
        "protocol": "intentir.cuda-dataset-compile-probe.v1",
        "dataset_root": str(catalog.root),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--frontend", choices=("cuda", "tilelang"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate", action="store_true")
    action.add_argument("--probe-tilelang", action="store_true")
    action.add_argument("--probe-cuda", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cuda-arch", default="sm_80")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    catalog = load_frontend_dataset(arguments.dataset_root)
    if arguments.probe_cuda:
        payload = _probe_cuda(
            catalog,
            workers=arguments.workers,
            timeout=arguments.timeout,
            arch=arguments.cuda_arch,
        )
    elif arguments.probe_tilelang:
        payload = _probe_tilelang(
            catalog,
            workers=arguments.workers,
            timeout=arguments.timeout,
        )
    elif arguments.validate:
        payload = catalog.validate().to_json()
    else:
        selected = catalog.select(frontend=arguments.frontend)
        payload = {
            "protocol": "intentir.multifrontend-dataset.v1",
            "root": str(catalog.root),
            "kernels": [item.to_json() for item in selected],
        }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if arguments.probe_tilelang or arguments.probe_cuda:
        return 0 if payload["failed"] == 0 else 2
    return 0 if not arguments.validate or payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
