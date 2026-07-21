"""Build and verify the offline receipt that precedes a formal LLM batch."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

from .lift_wire import verify_lift_wire_lock
from .real_batch import (
    BatchManifest,
    load_batch_manifest,
    probe_cute_compilation,
    probe_environment,
    probe_triton_compilation,
)


PREFLIGHT_SCHEMA = "intentir.real-llm-batch-preflight.v2"
_CODE_PATHS = (
    "python/intentir/evaluation/lift_wire.py",
    "python/intentir/evaluation/real_batch.py",
    "python/intentir/evaluation/formal_preflight.py",
    "python/intentir/evaluation/formal_runner.py",
    "python/intentir/verify/gate1/_triton_worker.py",
    "python/intentir/verify/gate1/compiler.py",
    "python/intentir/verify/gate1/ttir.py",
    "python/intentir/verify/gate1/codegen.py",
    "python/intentir/verify/gate1/orchestrate.py",
    "python/intentir/verify/gate1/_cute_worker.py",
    "python/intentir/verify/gate1/cute_compiler.py",
    "python/intentir/verify/gate1/cute.py",
    "python/intentir/verify/gate1/cute_codegen.py",
    "python/intentir/verify/gate1/cute_orchestrate.py",
    "python/intentir/verify/gate1/replay.py",
    "python/intentir/verify/gate1/prompt.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def current_gate1_code_identity(
    repository_root: str | Path | None = None,
) -> list[dict[str, str]]:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return [
        {"path": path, "sha256": _sha256((root / path).read_bytes())}
        for path in _CODE_PATHS
    ]


def _case_receipts(
    manifest: BatchManifest,
    triton_probe: dict[str, Any],
    cute_probe: dict[str, Any],
) -> list[dict[str, Any]]:
    triton = {
        item["case_id"]: item
        for item in triton_probe["cases"]
        if item["frontend"] == "triton"
    }
    cute = {item["case_id"]: item for item in cute_probe["cases"]}
    receipts: list[dict[str, Any]] = []
    for case in manifest.cases:
        case_id = case["case_id"]
        frontend = case["source"]["frontend"]
        if frontend == "triton":
            item = triton[case_id]
            receipt = {
                "case_id": case_id,
                "frontend": frontend,
                "status": item["status"],
                "source_observer_ready": bool(item["ok"]),
                "replay_arm_ready": bool(item["ok"]),
                "ttir_sha256": item["ttir_sha256"],
                "ttgir_sha256": item["ttgir_sha256"],
                "compile_cache_key": item["cache_key"],
            }
        else:
            item = cute[case_id]
            fingerprint = item.get("fingerprint") or {}
            async_operations = list(fingerprint.get("async_operations") or [])
            async_copy = next(
                (
                    operation
                    for operation in async_operations
                    if operation.startswith("cp.async.bulk")
                ),
                None,
            )
            receipt = {
                "case_id": case_id,
                "frontend": frontend,
                "status": item["status"],
                "source_observer_ready": bool(item["ok"]),
                "replay_arm_ready": bool(item["ok"]),
                "entry_symbol": item["entry_symbol"],
                "mlir_sha256": item["mlir_sha256"],
                "ptx_sha256": item["ptx_sha256"],
                "cubin_sha256": item["cubin_sha256"],
                "fingerprint_sha256": fingerprint.get("digest"),
                "normalized_mlir_sha256": fingerprint.get("mlir_sha256"),
                "ptx_opcode_sha256": fingerprint.get("ptx_opcode_sha256"),
                "async_copy_opcode": async_copy,
                "compile_cache_key": item["cache_key"],
            }
        if not item["ok"]:
            receipt["diagnostic"] = item["diagnostic"]
        receipts.append(receipt)
    return receipts


def build_formal_preflight(
    manifest_path: str | Path,
    *,
    source_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = 300.0,
    observed_date: str | None = None,
) -> dict[str, Any]:
    """Run all offline checks; this function contains no provider call path."""

    path = Path(manifest_path).resolve()
    manifest = load_batch_manifest(
        path,
        source_root=source_root,
        verify_sources=True,
        verify_repositories=True,
    )
    if "lift_wire" not in manifest.data:
        raise ValueError("formal preflight v2 requires a protocol-v2 manifest")
    wire = verify_lift_wire_lock(manifest.data["lift_wire"])
    triton_probe = probe_triton_compilation(
        manifest, timeout=timeout, cache_dir=cache_dir
    )
    cute_probe = probe_cute_compilation(
        manifest, timeout=timeout, cache_dir=cache_dir
    )
    cases = _case_receipts(manifest, triton_probe, cute_probe)
    source_ready = sum(bool(item["source_observer_ready"]) for item in cases)
    replay_ready = sum(bool(item["replay_arm_ready"]) for item in cases)
    environment = probe_environment(manifest)
    overall = bool(
        source_ready == len(cases)
        and replay_ready == len(cases)
        and environment["ready"]
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "batch_id": manifest.data["batch_id"],
        "manifest_schema": manifest.data["schema"],
        "manifest_digest": manifest.digest,
        "manifest_file_sha256": _sha256(path.read_bytes()),
        "observed_date": observed_date or date.today().isoformat(),
        "online_llm_calls": 0,
        "repository_revisions_verified": True,
        "lift_wire": wire,
        "gate1_code": current_gate1_code_identity(),
        "environment": environment,
        "summary": {
            "source_pins_verified": len(cases),
            "symbol_pins_verified": len(cases),
            "triton_total": triton_probe["triton_total"],
            "triton_compiled": triton_probe["triton_compiled"],
            "cute_total": cute_probe["cute_total"],
            "cute_compiled": cute_probe["cute_compiled"],
            "source_observer_ready": source_ready,
            "replay_arm_ready": replay_ready,
            "overall_ready": overall,
            "blocking_capability": None if overall else "formal_frontend_preflight",
        },
        "cases": cases,
    }


def verify_formal_preflight(
    receipt: dict[str, Any],
    manifest: BatchManifest,
    manifest_path: str | Path,
) -> None:
    """Fail closed when a stored v2 receipt no longer binds current code/data."""

    path = Path(manifest_path).resolve()
    expected_scalars = {
        "schema": PREFLIGHT_SCHEMA,
        "batch_id": manifest.data["batch_id"],
        "manifest_schema": manifest.data["schema"],
        "manifest_digest": manifest.digest,
        "manifest_file_sha256": _sha256(path.read_bytes()),
        "online_llm_calls": 0,
        "repository_revisions_verified": True,
    }
    for key, expected in expected_scalars.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal preflight {key} differs from the active epoch")
    if receipt.get("lift_wire") != verify_lift_wire_lock(
        manifest.data["lift_wire"]
    ):
        raise ValueError("formal preflight Lift wire identity drifted")
    if receipt.get("gate1_code") != current_gate1_code_identity():
        raise ValueError("formal preflight Gate1 code identity drifted")
    if receipt.get("environment") != probe_environment(manifest):
        raise ValueError("formal preflight environment identity drifted")
    summary = receipt.get("summary")
    if not isinstance(summary, dict) or summary.get("overall_ready") is not True:
        raise ValueError("formal preflight is not 28/28 ready")
    for key in (
        "source_pins_verified",
        "symbol_pins_verified",
        "source_observer_ready",
        "replay_arm_ready",
    ):
        if summary.get(key) != len(manifest.cases):
            raise ValueError(f"formal preflight {key} is incomplete")
    cases = receipt.get("cases")
    if not isinstance(cases, list) or [item.get("case_id") for item in cases] != [
        item["case_id"] for item in manifest.cases
    ]:
        raise ValueError("formal preflight case order differs from the manifest")
    if any(
        item.get("source_observer_ready") is not True
        or item.get("replay_arm_ready") is not True
        for item in cases
    ):
        raise ValueError("formal preflight contains an unready frontend arm")


__all__ = [
    "PREFLIGHT_SCHEMA",
    "build_formal_preflight",
    "current_gate1_code_identity",
    "verify_formal_preflight",
]
