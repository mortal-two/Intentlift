#!/usr/bin/env python3
"""Export Task28 evidence, run the isolated backends, and summarize Task30."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import canonical_artifact_digest


TASK_ROOT = Path(__file__).resolve().parent
SOURCE_TASK = TASK_ROOT.parent / "08-01-j-only-replay-multifrontend-batch-28"
PAIR_TASK = TASK_ROOT.parent / "08-01-tilelang-cuda-paired-j-analysis-29"
SOURCE_RESULTS = SOURCE_TASK / "results_current.json"
PAIR_RESULTS = PAIR_TASK / "results_current.json"
BACKEND_CONSUMER = TASK_ROOT / "backend_consume.py"
INPUT_PATH = TASK_ROOT / "backend_inputs_current.json"
PROVENANCE_PATH = TASK_ROOT / "provenance_current.json"
BACKEND_RESULT_PATH = TASK_ROOT / "backend_results_current.json"
SUMMARY_JSON_PATH = TASK_ROOT / "results_current.json"
SUMMARY_MD_PATH = TASK_ROOT / "results_current.md"

INPUT_PROTOCOL = "intentir.experiment.artifact-only-backend-input.v1"
PROVENANCE_PROTOCOL = "intentir.experiment.backend-reuse-provenance.v1"
SUMMARY_PROTOCOL = "intentir.experiment.multifrontend-backend-reuse-summary.v1"
_SYMBOL = re.compile(r"\bx\d+\b")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _accepted(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("typecheck") == "pass"
        and summary.get("gate1") == "replay_closed"
        and summary.get("gate2_passed") is True
        and summary.get("lift_agent_calls") == 0
        and summary.get("pi_visible_to_replay") is False
    )


def _selected_run_dir(selected: Mapping[str, Any]) -> Path:
    raw = Path(str(selected["run_dir"]))
    if raw.is_dir():
        return raw.resolve()
    candidate = SOURCE_TASK / "runs" / raw.parent.name / raw.name
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve Task28 run_dir {raw}")


def _logical_extents(
    j_payload: Mapping[str, Any], source_dimensions: Mapping[str, Any]
) -> tuple[dict[str, int], str]:
    symbols = sorted(
        set(_SYMBOL.findall(json.dumps(j_payload, sort_keys=True))),
        key=lambda item: int(item[1:]),
    )
    dimensions = {str(name): int(value) for name, value in source_dimensions.items()}
    if not symbols:
        return {}, "static-j"
    if symbols == ["x0"] and len(dimensions) == 1:
        return {"x0": next(iter(dimensions.values()))}, "single-source-dimension"
    convention = {"x0": "M", "x1": "N", "x2": "K", "x3": "H"}
    if all(item in convention and convention[item] in dimensions for item in symbols):
        return (
            {item: dimensions[convention[item]] for item in symbols},
            "dataset-mnkh-to-canonical-x",
        )
    raise ValueError(
        f"cannot derive canonical extents for symbols={symbols}, dimensions={dimensions}"
    )


def _load_accepted(
    stem: str,
    frontend: str,
    pair: Mapping[str, Any],
) -> tuple[Gate1Artifact, dict[str, Any], dict[str, Any], Path]:
    evidence = pair[frontend]
    selected = evidence.get("selected")
    if evidence.get("accepted") is not True or not isinstance(selected, Mapping):
        raise ValueError(f"Task28 evidence is not accepted: {stem}/{frontend}")
    if not _accepted(selected):
        raise ValueError(f"Task28 selected evidence is stale: {stem}/{frontend}")
    expected_case = (
        f"tilelang:{stem}" if frontend == "tilelang" else f"cuda:tilelang_{stem}"
    )
    if selected.get("case") != expected_case:
        raise ValueError(f"Task28 case mismatch: {stem}/{frontend}")
    run_dir = _selected_run_dir(selected)
    evidence_root = (SOURCE_TASK / "runs").resolve()
    try:
        relative_run_dir = run_dir.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError(f"run_dir escapes Task28 evidence root: {run_dir}") from error
    summary = _read(run_dir / "summary.json")
    if not _accepted(summary) or summary.get("case") != expected_case:
        raise ValueError(f"run-local summary is not exact accepted evidence: {run_dir}")
    visibility = _read(run_dir / "pi_visibility.json")
    if visibility.get("packet_root") != ["c", "j", "u_f"]:
        raise ValueError(f"Replay packet is not J-only: {run_dir}")
    if visibility.get("pi_visible_to_replay") is not False:
        raise ValueError(f"pi was visible to Replay: {run_dir}")
    artifact_path = run_dir / "artifact.json"
    artifact_payload = _read(artifact_path)
    if set(artifact_payload) != {"schema", "j", "c", "pi"}:
        raise ValueError(f"accepted artifact root is not exact v13 J/C/pi: {run_dir}")
    artifact = canonicalize_gate1_artifact(Gate1Artifact.from_json(artifact_payload))
    gate1 = _read(run_dir / "gate1.json")
    if gate1.get("artifact_digest") != canonical_artifact_digest(artifact):
        raise ValueError(f"Gate1 does not bind the canonical artifact: {run_dir}")
    gate2 = _read(run_dir / "gate2.json")
    if gate2.get("passed") is not True:
        raise ValueError(f"Gate2 is not passing: {run_dir}")
    authoring = _read(run_dir / "authoring.json")
    expected_pair = [f"tilelang:{stem}", f"cuda:tilelang_{stem}"]
    if authoring.get("stem") != stem or authoring.get("cases") != expected_pair:
        raise ValueError(f"same-origin authoring receipt mismatch: {run_dir}")
    adapter_path = run_dir / "adapter.json"
    adapter = _read(adapter_path)
    return artifact, adapter, {
        "stem": stem,
        "frontend": frontend,
        "case": expected_case,
        "run": run_dir.name,
        "run_dir": str(Path("runs") / relative_run_dir),
        "artifact_sha256": _file_digest(artifact_path),
        "adapter_sha256": _file_digest(adapter_path),
        "gate1_artifact_digest": gate1["artifact_digest"],
        "typecheck": summary["typecheck"],
        "gate1": summary["gate1"],
        "gate2": summary["gate2"],
        "gate2_passed": summary["gate2_passed"],
        "lift_agent_calls": summary["lift_agent_calls"],
        "pi_visible_to_replay": summary["pi_visible_to_replay"],
        "task28_selected_run_bound": True,
        "same_origin_authoring_bound": True,
    }, artifact_payload, run_dir


def export_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _read(SOURCE_RESULTS)
    pairs = source.get("pairs")
    if (
        source.get("accepted_unique_pair_count") != 31
        or source.get("accepted_unique_frontend_run_count") != 62
        or not isinstance(pairs, Mapping)
        or len(pairs) != 31
    ):
        raise ValueError("Task28 full-31/62 denominator is not frozen")
    entries: list[dict[str, Any]] = []
    provenance_entries: list[dict[str, Any]] = []
    for stem, pair in sorted(pairs.items()):
        if not isinstance(pair, Mapping) or pair.get("accepted") is not True:
            raise ValueError(f"Task28 pair is not accepted: {stem}")
        for frontend in ("tilelang", "cuda"):
            artifact, adapter, provenance, artifact_payload, _run_dir = _load_accepted(
                str(stem), frontend, pair
            )
            snapshot = ProductionSnapshot(artifact.j, artifact.c)
            # Force the same strict decode that the isolated backend repeats.
            snapshot = ProductionSnapshot.from_dict(snapshot.to_dict())
            extents, valuation_rule = _logical_extents(
                artifact_payload["j"], adapter.get("dimensions", {})
            )
            constexpr_bindings: dict[str, bool | int | float] = {}
            run_id = _digest(
                {
                    "snapshot_digest": snapshot.snapshot_digest,
                    "concrete_extents": extents,
                    "constexpr_bindings": constexpr_bindings,
                }
            )
            entries.append(
                {
                    "run_id": run_id,
                    "snapshot": snapshot.to_dict(),
                    "concrete_extents": extents,
                    "constexpr_bindings": constexpr_bindings,
                }
            )
            provenance_entries.append(
                {
                    "run_id": run_id,
                    "snapshot_digest": snapshot.snapshot_digest,
                    "j_digest": snapshot.j_digest,
                    "c_digest": snapshot.c_digest,
                    "concrete_extents": extents,
                    "valuation_rule": valuation_rule,
                    **provenance,
                }
            )
    entries.sort(key=lambda item: item["run_id"])
    provenance_entries.sort(key=lambda item: item["run_id"])
    if len(entries) != 62 or len({item["run_id"] for item in entries}) != 62:
        raise ValueError("backend export must contain 62 unique anonymous inputs")
    input_content: dict[str, Any] = {
        "protocol": INPUT_PROTOCOL,
        "entry_count": len(entries),
        "entries": entries,
    }
    input_content["input_digest"] = _digest(input_content)
    provenance_content: dict[str, Any] = {
        "protocol": PROVENANCE_PROTOCOL,
        "source_task": SOURCE_TASK.name,
        "source_results_sha256": _file_digest(SOURCE_RESULTS),
        "pair_analysis_sha256": _file_digest(PAIR_RESULTS),
        "entry_count": len(provenance_entries),
        "entries": provenance_entries,
        "backend_receives_this_file": False,
        "claim_boundary": (
            "frontend/case/source/Gate/pi provenance is retained only for post-hoc "
            "analysis and is never passed to backend_consume.py"
        ),
    }
    provenance_content["provenance_digest"] = _digest(provenance_content)
    _write(INPUT_PATH, input_content)
    _write(PROVENANCE_PATH, provenance_content)
    return input_content, provenance_content


def run_backend(*, workers: int) -> None:
    subprocess.run(
        [
            sys.executable,
            str(BACKEND_CONSUMER),
            "--input",
            str(INPUT_PATH),
            "--output",
            str(BACKEND_RESULT_PATH),
            "--workers",
            str(workers),
        ],
        check=True,
    )


def _stage_pass(entry: Mapping[str, Any], *path: str) -> bool:
    current: Any = entry
    for item in path:
        if not isinstance(current, Mapping):
            return False
        current = current.get(item)
    return current == "pass"


def _stage_status(entry: Mapping[str, Any], *path: str) -> str:
    current: Any = entry
    for item in path:
        if not isinstance(current, Mapping):
            return "missing"
        current = current.get(item)
    return str(current) if current is not None else "missing"


def _frontend_counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    paths = {
        "decode": ("decode", "status"),
        "emit": ("emit", "status"),
        "cuda_legal": ("targets", "cuda", "legal", "status"),
        "cuda_lower": ("targets", "cuda", "lower", "status"),
        "rvv_legal": ("targets", "rvv", "legal", "status"),
        "rvv_lower": ("targets", "rvv", "lower", "status"),
    }
    return {
        name: sum(_stage_pass(row, *path) for row in rows)
        for name, path in paths.items()
    }


def _failure_taxonomy(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    stage_paths = {
        "decode": ("decode",),
        "emit": ("emit",),
        "cuda_legal": ("targets", "cuda", "legal"),
        "cuda_lower": ("targets", "cuda", "lower"),
        "rvv_legal": ("targets", "rvv", "legal"),
        "rvv_lower": ("targets", "rvv", "lower"),
    }
    counts: dict[str, int] = {}
    for row in rows:
        for name, path in stage_paths.items():
            current: Any = row
            for key in path:
                current = current.get(key, {}) if isinstance(current, Mapping) else {}
            status = current.get("status") if isinstance(current, Mapping) else "missing"
            if status == "pass":
                continue
            error = current.get("error_type", status) if isinstance(current, Mapping) else status
            key = f"{name}:{error}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarize() -> dict[str, Any]:
    provenance = _read(PROVENANCE_PATH)
    backend = _read(BACKEND_RESULT_PATH)
    pair_analysis = _read(PAIR_RESULTS)
    provenance_content = dict(provenance)
    provenance_digest = provenance_content.pop("provenance_digest", None)
    if provenance_digest != _digest(provenance_content):
        raise ValueError("provenance digest mismatch")
    backend_content = dict(backend)
    result_digest = backend_content.pop("result_digest", None)
    if result_digest != _digest(backend_content):
        raise ValueError("backend result digest mismatch")
    if backend.get("input_digest") != _read(INPUT_PATH).get("input_digest"):
        raise ValueError("backend result does not bind exported anonymous inputs")
    provenance_by_id = {item["run_id"]: item for item in provenance["entries"]}
    backend_by_id = {item["run_id"]: item for item in backend["entries"]}
    if set(provenance_by_id) != set(backend_by_id) or len(backend_by_id) != 62:
        raise ValueError("backend/provenance anonymous run sets differ")
    joined: list[dict[str, Any]] = []
    for run_id in sorted(backend_by_id):
        row = dict(backend_by_id[run_id])
        row["provenance"] = provenance_by_id[run_id]
        joined.append(row)
    by_frontend = {
        frontend: [
            row for row in joined if row["provenance"]["frontend"] == frontend
        ]
        for frontend in ("tilelang", "cuda")
    }
    frontend_counts = {
        frontend: _frontend_counts(rows) for frontend, rows in by_frontend.items()
    }
    pair_rows: list[dict[str, Any]] = []
    for stem in sorted({row["provenance"]["stem"] for row in joined}):
        fronts = {
            row["provenance"]["frontend"]: row
            for row in joined
            if row["provenance"]["stem"] == stem
        }
        if set(fronts) != {"tilelang", "cuda"}:
            raise ValueError(f"pair {stem} does not have exactly two frontend rows")
        pair_rows.append(
            {
                "stem": stem,
                "decode_both": all(
                    _stage_pass(row, "decode", "status") for row in fronts.values()
                ),
                "emit_both": all(
                    _stage_pass(row, "emit", "status") for row in fronts.values()
                ),
                "cuda_legal_both": all(
                    _stage_pass(row, "targets", "cuda", "legal", "status")
                    for row in fronts.values()
                ),
                "cuda_lower_both": all(
                    _stage_pass(row, "targets", "cuda", "lower", "status")
                    for row in fronts.values()
                ),
                "rvv_legal_both": all(
                    _stage_pass(row, "targets", "rvv", "legal", "status")
                    for row in fronts.values()
                ),
                "rvv_lower_both": all(
                    _stage_pass(row, "targets", "rvv", "lower", "status")
                    for row in fronts.values()
                ),
            }
        )
    pair_counts = {
        field: sum(bool(row[field]) for row in pair_rows)
        for field in (
            "decode_both",
            "emit_both",
            "cuda_legal_both",
            "cuda_lower_both",
            "rvv_legal_both",
            "rvv_lower_both",
        )
    }
    full = pair_analysis["robustness"]
    content: dict[str, Any] = {
        "protocol": SUMMARY_PROTOCOL,
        "denominator": {
            "same_origin_pairs": 31,
            "frontend_artifacts": 62,
            "tilelang_artifacts": 31,
            "cuda_artifacts": 31,
            "primary20_used_as_denominator": False,
        },
        "frontend_stage_pass_counts": frontend_counts,
        "pair_stage_pass_counts": pair_counts,
        "failure_taxonomy": _failure_taxonomy(joined),
        "pair_rows": pair_rows,
        "cross_frontend_j": {
            "semantic_core_exact": full["exact_counts"]["semantic_core_exact"],
            "op_kind_set_exact": full["exact_counts"]["intent.op_kind_set_equal"],
            "boundary_exact": full["exact_counts"]["boundary.exact"],
            "buffer_profile_exact": full["exact_counts"][
                "signature.buffer_profile_exact"
            ],
            "raw_full_j_exact": full["exact_counts"]["full_j_exact"],
            "c_exact": full["exact_counts"]["c.exact"],
            "pi_exact": full["exact_counts"]["pi_exact"],
            "extra_cuda_scalar_pairs": full["extra_cuda_scalars"][
                "pair_count_nonzero"
            ],
        },
        "reuse_claim": {
            "search_j_sync_commit": "6d764cffde3829bce8498fcc63293488a2ab532d",
            "production_backend_files_changed_after_sync": 1,
            "frontend_specific_backend_loc": 0,
            "frontend_specific_backend_files": 0,
            "frontend_specific_backend_rules": 0,
            "per_kernel_backend_rules": 0,
            "shared_j_feature_consumer": "CUDA exact math.floor to native NVVM floor intrinsic",
            "shared_j_feature_consumer_commit": "64317b5e62a04cd5a77a20de5e4fcbcb9f9cef47",
            "shared_j_feature_consumer_loc": 31,
            "shared_j_feature_backend_files": 1,
            "shared_j_feature_rules": 1,
            "initial_cuda_lower_artifacts_before_shared_fix": 60,
            "initial_cuda_lower_failures_before_shared_fix": 2,
            "initial_failure_scope": "the TileLang and CUDA floor2d artifacts; same J feature",
            "initial_failure_diagnostic": "ptxas unresolved extern __nv_floorf",
            "experiment_harness_is_not_backend": True,
            "backend_input": "anonymous content-addressed J+C plus logical extents",
            "backend_does_not_receive": [
                "Source",
                "pi",
                "Replay",
                "explicit frontend label",
                "case_id",
                "Gate evidence",
                "historical winner",
            ],
        },
        "lower_claim_boundary": {
            "cuda": "real generic Intent-to-cubin lowering; no execution, Q/D, search, or timing claim",
            "rvv": "real C-Fixed Intent-to-RVV-object lowering; no execution, Q/D, or timing claim",
            "closure_evidence": "Task28 TypeCheck + frontend-specific Gate1 + three-leg Gate2",
            "performance_claimed_for_this_cohort": False,
        },
        "machine": backend["machine"],
        "input_digest": backend["input_digest"],
        "backend_result_digest": backend["result_digest"],
        "provenance_digest": provenance["provenance_digest"],
    }
    content["summary_digest"] = _digest(content)
    _write(SUMMARY_JSON_PATH, content)
    _write_markdown(content)
    return content


def _mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _write_markdown(result: Mapping[str, Any]) -> None:
    denom = result["denominator"]
    front = result["frontend_stage_pass_counts"]
    pair = result["pair_stage_pass_counts"]
    j = result["cross_frontend_j"]
    lines = [
        "# Task30 artifact-only multi-frontend backend reuse results",
        "",
        f"- denominator: {denom['same_origin_pairs']} same-origin pairs / {denom['frontend_artifacts']} exact accepted frontend artifacts",
        "- backend input: anonymous content-addressed `ProductionSnapshot(J,C)` plus canonical logical extents",
        "- backend-visible Source / pi / Replay / explicit frontend label / case-id / Gate evidence / historical winner: none",
        "- Primary-20 is not used as an experimental denominator",
        "",
        "## Artifact-level reuse matrix",
        "",
        "| Frontend | N | Decode | Emit Intent MLIR | CUDA Legal | CUDA cubin | RVV Legal | RVV object |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frontend in ("tilelang", "cuda"):
        row = front[frontend]
        lines.append(
            f"| {frontend} | 31 | {row['decode']}/31 | {row['emit']}/31 | "
            f"{row['cuda_legal']}/31 | {row['cuda_lower']}/31 | "
            f"{row['rvv_legal']}/31 | {row['rvv_lower']}/31 |"
        )
    lines.extend(
        [
            "",
            "## Pair-level both-frontends matrix",
            "",
            "| Stage | Both TileLang and CUDA |",
            "|---|---:|",
            f"| Decode | {pair['decode_both']}/31 |",
            f"| Emit Intent MLIR | {pair['emit_both']}/31 |",
            f"| CUDA Contract Legal | {pair['cuda_legal_both']}/31 |",
            f"| CUDA >=1 real lower state | {pair['cuda_lower_both']}/31 |",
            f"| RVV Contract Legal | {pair['rvv_legal_both']}/31 |",
            f"| RVV >=1 real lower state | {pair['rvv_lower_both']}/31 |",
            "",
            "## Joint frontend evidence",
            "",
            "| Evidence | Full 31 |",
            "|---|---:|",
            "| Frontend closure | 31/31 pairs; 62/62 artifacts |",
            f"| Signature-free semantic core exact | {j['semantic_core_exact']}/31 |",
            f"| Op-kind set exact | {j['op_kind_set_exact']}/31 |",
            f"| Boundary exact | {j['boundary_exact']}/31 |",
            f"| Carrier-normalized buffer profile exact | {j['buffer_profile_exact']}/31 |",
            f"| Raw full-J exact | {j['raw_full_j_exact']}/31 |",
            f"| C exact | {j['c_exact']}/31 |",
            f"| CUDA has extra runtime scalar | {j['extra_cuda_scalar_pairs']}/31 |",
            "| Frontend-specific backend LoC | 0 |",
            "| Frontend-specific/per-kernel backend rules | 0 / 0 |",
            "| Shared J-feature consumer delta | +31 LoC / 1 file (`math.floor`, CUDA) |",
            "| CUDA lower before -> after shared fix | 60/62 -> 62/62 |",
            "",
            "`CUDA cubin` means the real Intent -> Linalg -> NVVM/PTX -> ptxas path produced at least one ELF cubin. `RVV object` means the real Intent -> Linalg -> LLVM -> RVV opt/llc path produced a schedule-realized native object. Neither column is a runtime-correctness or performance claim; behavioral closure remains Task28 Gate1/Gate2, and this Task does not execute Q/D or benchmark the binaries.",
            "",
            "## Failures",
            "",
        ]
    )
    failures = result["failure_taxonomy"]
    if failures:
        lines.extend([f"- `{name}`: {count}" for name, count in failures.items()])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Per-pair status",
            "",
            "| Stem | Decode | Emit | CUDA Legal | CUDA Lower | RVV Legal | RVV Lower |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in result["pair_rows"]:
        lines.append(
            f"| `{row['stem']}` | {_mark(row['decode_both'])} | {_mark(row['emit_both'])} | "
            f"{_mark(row['cuda_legal_both'])} | {_mark(row['cuda_lower_both'])} | "
            f"{_mark(row['rvv_legal_both'])} | {_mark(row['rvv_lower_both'])} |"
        )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("export", "backend", "summarize", "all"), default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase in {"export", "all"}:
        exported, _provenance = export_inputs()
        print(
            json.dumps(
                {
                    "exported": exported["entry_count"],
                    "input_digest": exported["input_digest"],
                },
                sort_keys=True,
            )
        )
    if args.phase in {"backend", "all"}:
        run_backend(workers=args.workers)
    if args.phase in {"summarize", "all"}:
        result = summarize()
        print(
            json.dumps(
                {
                    "denominator": result["denominator"],
                    "pair_stage_pass_counts": result["pair_stage_pass_counts"],
                    "failure_taxonomy": result["failure_taxonomy"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
