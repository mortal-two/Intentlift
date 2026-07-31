#!/usr/bin/env python3
"""Generate auditable J-only ``S_J`` evidence from one real SCRC artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.replay_compiler import build_semantic_program_mount
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.prompt import build_rewriter_packet
from intentir.verify.typecheck import type_check


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freshly materialize parameterized S_J from exact accepted J"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frontend",
        choices=[Frontend.TRITON.value, Frontend.CUDA.value, Frontend.TILELANG.value],
        default=Frontend.TILELANG.value,
    )
    parser.add_argument(
        "--mode", choices=["normal", "masked", "both"], default="normal"
    )
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be absent or empty; evidence is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    artifact_payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    artifact = canonicalize_gate1_artifact(
        Gate1Artifact.from_json(artifact_payload)
    )
    report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.requires_program_axes,
    )
    _write_json(args.out / "typecheck.json", report.to_json())
    if not report.passed:
        raise SystemExit("TypeCheck(J,C,pi) failed; S_J evidence was not generated")
    shutil.copyfile(args.artifact, args.out / "artifact.json")

    requested = (
        (False, True)
        if args.mode == "both"
        else (args.mode == "masked",)
    )
    arms: list[dict[str, Any]] = []
    frontend = Frontend(args.frontend)
    for masked in requested:
        # build_rewriter_packet independently canonicalizes, TypeChecks, and
        # creates the exact normal or MaskJ(J) source-isolated view.  S_J sees
        # only the resulting packet.j mapping.
        packet = build_rewriter_packet(
            artifact,
            frontend=frontend,
            masked=masked,
        )
        mount = build_semantic_program_mount(packet.j, masked=masked)
        arm_name = "masked" if masked else "normal"
        arm_dir = args.out / arm_name
        arm_dir.mkdir()
        for name, content in mount.files().items():
            (arm_dir / name).write_text(content, encoding="utf-8")
        arm = {
            "arm": arm_name,
            "frontend": frontend.value,
            "receipt": mount.receipt(),
        }
        _write_json(arm_dir / "report.json", arm)
        arms.append(arm)

    result = {
        "protocol": "intentir.scrc.semantic-materialization-evidence.v1",
        "status": "pass",
        "artifact_schema": artifact.wire_schema,
        "typecheck": report.to_json(),
        "arms": arms,
    }
    _write_json(args.out / "report.json", result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
