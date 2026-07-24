#!/usr/bin/env python3
"""Diagnostic-only Gate1 recheck for one already saved SCRC artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.extraction.providers.provider import get_provider
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.scrc_workload_run import (
    GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
    initialize_workload_gate1_recheck_evidence,
    next_gate1_recheck_dir,
    recheck_prepared_workload_gate1,
    save_workload_gate1_recheck,
    validate_workload_gate1_recheck_parent,
)
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="one active workload case id")
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="saved artifact.json to recheck without Lift",
    )
    parser.add_argument(
        "--parent-evidence",
        required=True,
        type=Path,
        help="immutable prior run/attempt directory containing the artifact",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/intentir-scrc-task11/fixed-gate1-rechecks"),
        help="separate evidence root; creates an atomic recheck-NNN directory",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def _emit(value: dict, *, output: Path | None, json_mode: bool) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if json_mode or output is None:
        print(encoded, end="")


def _load_parented_artifact(
    artifact_path: Path,
    parent_path: Path,
) -> tuple[Gate1Artifact, Path, Path]:
    parent = parent_path.expanduser().resolve()
    artifact_file = artifact_path.expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent evidence directory does not exist")
    if not artifact_file.is_file():
        raise ValueError("artifact file does not exist")
    try:
        artifact_file.relative_to(parent)
    except ValueError as exc:
        raise ValueError("artifact must be inside parent evidence") from exc
    try:
        payload = json.loads(artifact_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact JSON cannot be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact JSON root must be an object")
    try:
        artifact = Gate1Artifact.from_json(payload)
    except Exception as exc:
        raise ValueError(f"artifact decode rejected: {type(exc).__name__}: {exc}") from exc
    return artifact, artifact_file, parent


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        artifact, artifact_file, parent = _load_parented_artifact(
            args.artifact, args.parent_evidence
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    adapter = TRITON_WORKLOAD_ADAPTERS.get(args.case)
    if adapter is None:
        print("case has no active Triton workload adapter", file=sys.stderr)
        return 2
    try:
        prepared = prepare_triton_workload(
            adapter,
            dataset_root=args.dataset_root,
        )
        validate_workload_gate1_recheck_parent(parent, prepared, artifact)
        provider = get_provider(args.provider)
        recheck_dir = next_gate1_recheck_dir(args.run_root, args.case)
        initialize_workload_gate1_recheck_evidence(
            recheck_dir,
            prepared,
            provider=provider,
            effort=args.effort,
            parent_evidence=parent,
        )
        result = recheck_prepared_workload_gate1(
            prepared,
            artifact,
            provider=provider,
            effort=args.effort,
        )
        save_workload_gate1_recheck(recheck_dir, prepared, result)
    except Exception as exc:
        print(f"fixed-artifact Gate1 recheck failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    status = (
        "typecheck_failed"
        if result.gate1 is None
        else result.gate1.status.value
    )
    payload = {
        "protocol": GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
        "diagnostic_only": True,
        "enters_gate2": False,
        "changes_prior_verdict": False,
        "case_id": args.case,
        "artifact": str(artifact_file),
        "parent_evidence": str(parent),
        "evidence_path": str(recheck_dir),
        "artifact_digest": result.artifact_digest,
        "typecheck_status": result.typecheck.status.value,
        "status": status,
    }
    _emit(payload, output=args.output, json_mode=args.json)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
