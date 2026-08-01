"""Run the real J-only Replay/Gate workflow for one manually authored add2d case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from manual_add2d_artifacts import (
    cuda_add2d_artifact,
    paired_authoring_receipt,
    tilelang_add2d_artifact,
)
from intentir.extraction.providers.provider import get_provider
from intentir.verify.adapters import NativeGate2ExecutionAdapter
from intentir.verify.gate1.cuda_compiler import CudaCompiler
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.replay import (
    ReplayAttempt,
    ReplayGeneration,
    _replay_input_files,
)
from intentir.verify.gate2.native import CudaRunner, TileLangRunner
from intentir.verify.gate2.orchestrate import Gate2Runner
from intentir.verify.frontend_workloads import build_frontend_dataset_gate2_suite
from intentir.verify.scrc_workload_run import recheck_prepared_workload_gate1
from intentir.verify.workload_registry import prepare_registered_scrc_workload


CASE_BUILDERS = {
    "tilelang:add2d": tilelang_add2d_artifact,
    "cuda:tilelang_add2d": cuda_add2d_artifact,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _next_run_dir(root: Path, case_id: str) -> Path:
    case_dir = root / case_id.replace(":", "_")
    case_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("run-"))
        for path in case_dir.glob("run-[0-9][0-9][0-9]")
        if path.name.removeprefix("run-").isdigit()
    ]
    run_dir = case_dir / f"run-{max(existing, default=0) + 1:03d}"
    run_dir.mkdir()
    return run_dir


def _pi_key_paths(value: Any, path: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "pi":
                result.append(f"{path}.pi")
            result.extend(_pi_key_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_pi_key_paths(item, f"{path}[{index}]"))
    return result


def _replay_record(record: ReplayGeneration | ReplayAttempt) -> dict[str, Any]:
    common = {
        "kind": "generation" if isinstance(record, ReplayGeneration) else "attempt",
        "model": record.model,
        "response_model": record.response_model,
        "response_id": record.response_id,
        "backend_protocol": record.backend_protocol,
        "attempts": record.attempts,
        "physical_provider_calls": record.physical_provider_calls,
        "transport_retries": record.transport_retries,
        "elapsed_ms": record.elapsed_ms,
        "usage": dict(record.usage),
        "trajectory": dict(record.trajectory),
    }
    if isinstance(record, ReplayGeneration):
        common.update(
            {
                "function_name": record.function_name,
                "entry_function_name": record.entry_function_name,
                "context_receipt": dict(record.context_receipt),
            }
        )
    else:
        common["diagnostic"] = record.diagnostic
    return common


def _execute_gate2(prepared: Any, artifact: Gate1Artifact, gate1: Any) -> Any:
    if prepared.request.frontend == "cuda":
        source_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
        replay_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
    else:
        source_runner = TileLangRunner()
        replay_runner = TileLangRunner()
    adapter = NativeGate2ExecutionAdapter(
        prepared.request.frontend,
        Gate2Runner(source_runner=source_runner, replay_runner=replay_runner),
        lambda accepted, _gate1, _binding: build_frontend_dataset_gate2_suite(
            prepared, accepted
        ),
    )
    return adapter.execute(artifact, gate1, prepared.request)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    args = parser.parse_args()

    run_dir = _next_run_dir(args.run_root, args.case)
    started = time.time()
    provider = get_provider(args.provider)
    artifact = Gate1Artifact.from_json(CASE_BUILDERS[args.case]())
    prepared = prepare_registered_scrc_workload(args.case)
    frontend = Frontend(prepared.request.frontend)
    files, packet, _contract = _replay_input_files(
        artifact, frontend=frontend, masked=False
    )
    pi_paths = _pi_key_paths(packet)
    visibility = {
        "protocol": "intentir.experiment.replay-input-visibility.v1",
        "packet_root": sorted(packet),
        "workspace_files": sorted(files),
        "pi_key_paths": pi_paths,
        "pi_visible_to_replay": not not pi_paths,
        "source_visible_to_replay": False,
        "gate_visible_to_replay": False,
        "packet_sha256": hashlib.sha256(files["packet.json"].encode()).hexdigest(),
    }
    if pi_paths or set(packet) != {"j", "c", "u_f"}:
        raise RuntimeError("Replay visibility audit found a non-J-only packet")

    run_dir.joinpath("source.py").write_text(
        prepared.request.source.text, encoding="utf-8"
    )
    _write_json(run_dir / "adapter.json", prepared.adapter.to_json())
    _write_json(run_dir / "artifact.json", artifact.to_json())
    _write_json(run_dir / "authoring.json", paired_authoring_receipt())
    _write_json(run_dir / "pi_visibility.json", visibility)
    _write_json(
        run_dir / "request.json",
        {
            "case": args.case,
            "frontend": frontend.value,
            "provider": provider.name,
            "model": provider.model,
            "effort": args.effort,
            "lift_agent_calls": 0,
            "artifact_author": "primary Codex agent",
            "replay_backend": "codex",
            "run_masked_j": False,
        },
    )

    gate1_started = time.time()
    recheck = recheck_prepared_workload_gate1(
        prepared,
        artifact,
        provider=provider,
        effort=args.effort,
        replay_backend="codex",
        replay_composer_mode="off",
        run_masked_j=False,
    )
    gate1_elapsed_ms = (time.time() - gate1_started) * 1000.0
    _write_json(run_dir / "typecheck.json", recheck.typecheck.to_json())
    if recheck.gate1 is not None:
        _write_json(run_dir / "gate1.json", recheck.gate1.to_json())
    _write_json(
        run_dir / "timing.json",
        {
            "gate1_elapsed_ms": gate1_elapsed_ms,
            "gate2_elapsed_ms": 0.0,
            "total_elapsed_ms": (time.time() - started) * 1000.0,
        },
    )

    records = recheck.replay_writer.records if recheck.replay_writer else ()
    _write_json(run_dir / "replay_records.json", [_replay_record(row) for row in records])
    generations = [row for row in records if isinstance(row, ReplayGeneration)]
    if generations:
        run_dir.joinpath("replay.source").write_text(
            generations[-1].code, encoding="utf-8"
        )
        _write_json(
            run_dir / "replay_receipt.json",
            dict(generations[-1].context_receipt),
        )
        _write_json(
            run_dir / "replay_trajectory.json",
            dict(generations[-1].trajectory),
        )

    gate2 = None
    gate2_elapsed_ms = 0.0
    if recheck.passed:
        gate2_started = time.time()
        gate2 = _execute_gate2(prepared, recheck.artifact, recheck.gate1)
        gate2_elapsed_ms = (time.time() - gate2_started) * 1000.0
        _write_json(run_dir / "gate2.json", gate2.to_json())
        _write_json(
            run_dir / "timing.json",
            {
                "gate1_elapsed_ms": gate1_elapsed_ms,
                "gate2_elapsed_ms": gate2_elapsed_ms,
                "total_elapsed_ms": (time.time() - started) * 1000.0,
            },
        )

    gate2_status = None
    gate2_status_counts: dict[str, int] = {}
    if gate2 is not None:
        gate2_status = "gate2_pass" if gate2.passed else "gate2_fail"
        gate2_status_counts = dict(gate2.status_counts)

    summary = {
        "protocol": "intentir.experiment.manual-add2d-j-only.v1",
        "case": args.case,
        "frontend": frontend.value,
        "typecheck": recheck.typecheck.status.value,
        "gate1": recheck.gate1.status.value if recheck.gate1 else None,
        "gate2": gate2_status,
        "gate2_passed": gate2.passed if gate2 is not None else None,
        "gate2_status_counts": gate2_status_counts,
        "pi_visible_to_replay": False,
        "lift_agent_calls": 0,
        "replay_provider_calls": sum(row.physical_provider_calls for row in records),
        "gate1_elapsed_ms": gate1_elapsed_ms,
        "gate2_elapsed_ms": gate2_elapsed_ms,
        "total_elapsed_ms": (time.time() - started) * 1000.0,
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0 if recheck.passed and gate2 is not None and gate2.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
