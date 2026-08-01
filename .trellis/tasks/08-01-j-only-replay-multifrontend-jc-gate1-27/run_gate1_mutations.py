"""Run controlled add2d mutations through the real native Gate1 observers.

This is experiment-only sensitivity evidence.  It neither calls Lift nor asks a
Replay model to generate code: ``StaticReplayWriter`` transports one exact,
explicitly mutated copy of an already certified Replay into the ordinary native
compiler/observer/comparator pipeline.  Gate2 is deliberately unavailable here.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate1.cuda_compiler import CudaCompiler
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.native_orchestrate import (
    CudaGate1Runner,
    TileLangGate1Runner,
)
from intentir.verify.gate1.orchestrate import Gate1Config
from intentir.verify.gate1.replay import StaticReplayWriter
from intentir.verify.gate1.tilelang_compiler import TileLangCompiler
from intentir.verify.typecheck import type_check
from intentir.verify.workload_registry import prepare_registered_scrc_workload


TASK_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = TASK_ROOT / "mutations" / "add2d-v1"

CASES: dict[str, dict[str, Any]] = {
    "tilelang": {
        "case_id": "tilelang:add2d",
        "accepted": TASK_ROOT / "runs" / "tilelang_add2d" / "run-002",
        "frontend": Frontend.TILELANG,
        "mutations": {
            "wrong_op": (
                "p0[v1, v2] + p1[v1, v2]",
                "p0[v1, v2] - p1[v1, v2]",
                "replace the J-owned add with subtract",
            ),
            "wrong_index": (
                "p1[v1, v2]",
                "p1[v1, 0]",
                "rewire the second input from the current column to column zero",
            ),
            "wrong_effect": (
                "p2[v1, v2] = p0[v1, v2] + p1[v1, v2]",
                "p2[v1, v2] += p0[v1, v2] + p1[v1, v2]",
                "change a pure output write into read-modify-write",
            ),
        },
        "probes": {},
    },
    "cuda": {
        "case_id": "cuda:tilelang_add2d",
        "accepted": TASK_ROOT / "runs" / "cuda_tilelang_add2d" / "run-002",
        "frontend": Frontend.CUDA,
        "mutations": {
            "wrong_op": (
                "p0[index] + p1[index]",
                "p0[index] - p1[index]",
                "replace the J-owned add with subtract",
            ),
            "wrong_index": (
                "threadIdx.x",
                "threadIdx.y",
                "rewire the element coordinate from the x lane to the y lane",
            ),
            "wrong_effect": (
                "p2[index] = p0[index] + p1[index];",
                "p2[index] += p0[index] + p1[index];",
                "change a pure output write into read-modify-write",
            ),
        },
        "probes": {
            "address_offset_probe": (
                "p1[index]",
                "p1[index + 1ULL]",
                "probe whether the current semantic PTX observer retains a one-element address offset",
            )
        },
    },
}


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


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"controlled mutation expected one match, observed {count}: {old!r}")
    return source.replace(old, new, 1)


def _run_gate1(
    *,
    frontend: Frontend,
    prepared: Any,
    artifact: Gate1Artifact,
    replay_source: str,
) -> tuple[Any, float]:
    writer = StaticReplayWriter(
        replay_source,
        model="controlled-mutation-writer",
        frontend=frontend,
    )
    config = Gate1Config(run_masked_j=False)
    if frontend is Frontend.CUDA:
        runner: Any = CudaGate1Runner(
            compiler=CudaCompiler(timeout=240.0),
            writer=writer,
            config=config,
        )
    else:
        runner = TileLangGate1Runner(
            compiler=TileLangCompiler(timeout=240.0),
            writer=writer,
            config=config,
        )
    started = time.perf_counter()
    result = runner.run(
        artifact,
        source=prepared.request.source.gate_source,
        source_case=prepared.request.binding.gate1_case,
    )
    return result, (time.perf_counter() - started) * 1000.0


def _case() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frontend_name, spec in CASES.items():
        accepted = Path(spec["accepted"])
        artifact = canonicalize_gate1_artifact(
            Gate1Artifact.from_json(
                json.loads(accepted.joinpath("artifact.json").read_text(encoding="utf-8"))
            )
        )
        checked = type_check(
            artifact.j,
            artifact.c,
            artifact.pi,
            require_program_axes=artifact.requires_program_axes,
        )
        if not checked.passed:
            raise RuntimeError(f"accepted {frontend_name} artifact no longer TypeChecks")
        prepared = prepare_registered_scrc_workload(str(spec["case_id"]))
        accepted_source = accepted.joinpath("replay.source").read_text(encoding="utf-8")
        frontend = spec["frontend"]

        variants: dict[str, tuple[str, str | None, str]] = {
            "baseline": (accepted_source, None, "baseline")
        }
        for mutation, (old, new, description) in spec["mutations"].items():
            variants[mutation] = (
                _replace_once(accepted_source, old, new),
                description,
                "mutation",
            )
        for probe, (old, new, description) in spec["probes"].items():
            variants[probe] = (
                _replace_once(accepted_source, old, new),
                description,
                "coverage_probe",
            )

        for mutation, (replay_source, description, role) in variants.items():
            evidence_dir = OUTPUT_ROOT / frontend_name / mutation
            evidence_dir.mkdir(parents=True, exist_ok=False)
            evidence_dir.joinpath("replay.source").write_text(
                replay_source, encoding="utf-8"
            )
            result, elapsed_ms = _run_gate1(
                frontend=frontend,
                prepared=prepared,
                artifact=artifact,
                replay_source=replay_source,
            )
            gate = result.to_json()
            comparison = gate.get("normal", {}).get("comparison")
            hard_passes = (
                comparison.get("hard_passes", {})
                if isinstance(comparison, dict)
                else {}
            )
            failed_hard_fields = sorted(
                key for key, value in hard_passes.items() if value is False
            )
            compile_value = gate.get("normal", {}).get("compile")
            replay_compile_ok = (
                compile_value.get("ok")
                if isinstance(compile_value, dict)
                else False
            )
            row = {
                "frontend": frontend_name,
                "case": spec["case_id"],
                "mutation": mutation,
                "role": role,
                "description": description,
                "parent_accepted_run": str(accepted.relative_to(TASK_ROOT)),
                "typecheck": checked.status.value,
                "gate1_status": gate["status"],
                "gate1_elapsed_ms": elapsed_ms,
                "replay_compile_ok": replay_compile_ok,
                "hard_pass": (
                    comparison.get("hard_pass")
                    if isinstance(comparison, dict)
                    else None
                ),
                "primary_difference_category": (
                    comparison.get("primary_difference_category")
                    if isinstance(comparison, dict)
                    else None
                ),
                "failed_hard_fields": failed_hard_fields,
                "gate2_run": False,
                "lift_agent_calls": 0,
                "replay_provider_calls": 0,
                "writer": "exact controlled mutation transport",
            }
            _write_json(
                evidence_dir / "mutation.json",
                {
                    "protocol": "intentir.experiment.controlled-replay-mutation.v1",
                    "diagnostic_only": True,
                    "enters_gate2": False,
                    "description": description,
                    "parent_accepted_run": row["parent_accepted_run"],
                    "mutation": mutation,
                    "role": role,
                },
            )
            _write_json(evidence_dir / "gate1.json", gate)
            _write_json(evidence_dir / "summary.json", row)
            if role == "baseline":
                if gate["status"] != "replay_closed":
                    raise RuntimeError(f"{frontend_name} accepted baseline no longer closes")
            elif role == "mutation" and (
                gate["status"] != "no_closure"
                or not replay_compile_ok
                or row["hard_pass"] is not False
            ):
                raise RuntimeError(
                    f"{frontend_name}/{mutation} did not produce a compiled hard rejection"
                )
            rows.append(row)
    return rows


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"mutation evidence already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    rows = _case()
    mutations = [row for row in rows if row["role"] == "mutation"]
    probes = [row for row in rows if row["role"] == "coverage_probe"]
    report = {
        "protocol": "intentir.experiment.native-gate1-mutation-sensitivity.v1",
        "status": (
            "pass_with_known_coverage_gap"
            if any(row["hard_pass"] is True for row in probes)
            else "pass"
        ),
        "diagnostic_only": True,
        "lift_agent_calls": 0,
        "replay_provider_calls": 0,
        "gate2_calls": 0,
        "baseline_closed": sum(
            row["role"] == "baseline" and row["gate1_status"] == "replay_closed"
            for row in rows
        ),
        "mutation_count": len(mutations),
        "compiled_mutations": sum(bool(row["replay_compile_ok"]) for row in mutations),
        "gate1_hard_rejections": sum(row["hard_pass"] is False for row in mutations),
        "coverage_probe_count": len(probes),
        "coverage_probe_hard_rejections": sum(
            row["hard_pass"] is False for row in probes
        ),
        "rows": rows,
    }
    _write_json(OUTPUT_ROOT / "mutation_metrics.json", report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
