"""Run Task28 manual artifacts through the real J-only Replay/Gate workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

from manual_g1_artifacts import (
    CASE_BUILDERS as G1_CASE_BUILDERS,
    CASE_IDS as G1_CASE_IDS,
    CASE_RECEIPTS as G1_CASE_RECEIPTS,
)
from manual_g2_artifacts import (
    CASE_BUILDERS as G2_CASE_BUILDERS,
    CASE_IDS as G2_CASE_IDS,
    CASE_RECEIPTS as G2_CASE_RECEIPTS,
)
from manual_g3_g7_artifacts import (
    CASE_BUILDERS as G3_G7_CASE_BUILDERS,
    CASE_IDS as G3_G7_CASE_IDS,
    CASE_RECEIPTS as G3_G7_CASE_RECEIPTS,
)
from manual_descriptor_artifacts import (
    CASE_BUILDERS as DESCRIPTOR_CASE_BUILDERS,
    CASE_IDS as DESCRIPTOR_CASE_IDS,
    CASE_RECEIPTS as DESCRIPTOR_CASE_RECEIPTS,
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
from intentir.verify.gate2.policy import gate2_policy_for
from intentir.verify.frontend_workloads import build_frontend_dataset_gate2_suite
from intentir.verify.scrc_workload_run import recheck_prepared_workload_gate1
from intentir.verify.typecheck import type_check
from intentir.verify.workload_registry import prepare_registered_scrc_workload


TASK_ROOT = Path(__file__).resolve().parent
CASE_IDS = (*G1_CASE_IDS, *G2_CASE_IDS, *G3_G7_CASE_IDS, *DESCRIPTOR_CASE_IDS)
CASE_BUILDERS = {
    **G1_CASE_BUILDERS,
    **G2_CASE_BUILDERS,
    **G3_G7_CASE_BUILDERS,
    **DESCRIPTOR_CASE_BUILDERS,
}
CASE_RECEIPTS = {
    **G1_CASE_RECEIPTS,
    **G2_CASE_RECEIPTS,
    **G3_G7_CASE_RECEIPTS,
    **DESCRIPTOR_CASE_RECEIPTS,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _next_run_dir(root: Path, case_id: str) -> Path:
    """Reserve one run directory atomically without a scan-then-mkdir race."""

    case_dir = root / case_id.replace(":", "_")
    case_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000_000):
        run_dir = case_dir / f"run-{index:03d}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"exhausted run directory space for {case_id}")


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


def _visibility(artifact: Gate1Artifact, frontend: Frontend) -> dict[str, Any]:
    files, packet, _contract = _replay_input_files(
        artifact, frontend=frontend, masked=False
    )
    pi_paths = _pi_key_paths(packet)
    receipt = {
        "protocol": "intentir.experiment.replay-input-visibility.v1",
        "packet_root": sorted(packet),
        "workspace_files": sorted(files),
        "pi_key_paths": pi_paths,
        "pi_visible_to_replay": bool(pi_paths),
        "source_visible_to_replay": False,
        "gate_visible_to_replay": False,
        "packet_sha256": hashlib.sha256(files["packet.json"].encode()).hexdigest(),
    }
    if pi_paths or set(packet) != {"j", "c", "u_f"}:
        raise RuntimeError("Replay visibility audit found a non-J-only packet")
    return receipt


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


def _execute_gate2_inline(
    prepared: Any,
    artifact: Gate1Artifact,
    gate1: Any,
) -> Any:
    if prepared.request.frontend == "cuda":
        source_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
        replay_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
    else:
        source_runner = TileLangRunner()
        replay_runner = TileLangRunner()
    adapter = NativeGate2ExecutionAdapter(
        prepared.request.frontend,
        Gate2Runner(
            source_runner=source_runner,
            replay_runner=replay_runner,
            comparison_policy=gate2_policy_for(prepared.entry.q_policy),
        ),
        lambda accepted, _gate1, _binding: build_frontend_dataset_gate2_suite(
            prepared, accepted
        ),
    )
    return adapter.execute(artifact, gate1, prepared.request)


class _Gate2JsonResult:
    """Small parent-side view of a child Gate2 JSON certificate."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    @property
    def passed(self) -> bool:
        return bool(self.payload.get("passed", False))

    @property
    def status_counts(self) -> dict[str, int]:
        raw = self.payload.get("status_counts", {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    def to_json(self) -> dict[str, Any]:
        return dict(self.payload)


def _gate2_child_entry(
    connection: Any,
    case_id: str,
    artifact_json: Mapping[str, Any],
    replay_code: str,
    replay_function_name: str,
) -> None:
    """Execute one real Gate2 suite in a clean spawned interpreter.

    The child intentionally reconstructs the registered Source/Harness from the
    case id and receives only the exact J artifact plus the Gate1-accepted Replay
    text.  No Gate2 result, compiler context, or CUDA runtime is shared with the
    parent or another case.
    """

    try:
        artifact = Gate1Artifact.from_json(dict(artifact_json))
        prepared = prepare_registered_scrc_workload(case_id)
        generation = SimpleNamespace(
            code=replay_code,
            function_name=replay_function_name,
        )
        gate1 = SimpleNamespace(normal=SimpleNamespace(generation=generation))
        result = _execute_gate2_inline(prepared, artifact, gate1)
        connection.send({"ok": True, "result": result.to_json()})
    except BaseException as exc:  # propagate process failures as evidence
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def _execute_gate2(
    case_id: str,
    artifact: Gate1Artifact,
    gate1: Any,
    semaphore: threading.Semaphore,
) -> _Gate2JsonResult:
    """Run Gate2 in an isolated ``spawn`` process per experiment case.

    A CUDA illegal-address event poisons the owning context.  Sharing the old
    inline context made all later cases report driver error 700, so the formal
    batch must isolate cases at the process boundary.  ``spawn`` is deliberate:
    ``fork`` can inherit a partially initialized CUDA runtime.
    """

    generation = getattr(getattr(gate1, "normal", None), "generation", None)
    if generation is None:
        raise ValueError("Gate2 isolation requires the exact Gate1-accepted Replay")
    with semaphore:
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(False)
        process = context.Process(
            target=_gate2_child_entry,
            args=(
                send,
                case_id,
                artifact.to_json(),
                generation.code,
                generation.function_name,
            ),
            name=f"intentir-gate2-{case_id.replace(':', '-')}",
        )
        process.start()
        send.close()
        timeout = 600.0
        message: Mapping[str, Any]
        try:
            if not receive.poll(timeout):
                process.terminate()
                process.join(10.0)
                return _Gate2JsonResult(
                    {
                        "suite_id": case_id,
                        "passed": False,
                        "failed_case_ids": ["gate2_process_timeout"],
                        "status_counts": {"process_timeout": 1},
                        "cases": [],
                        "metadata": {"error": f"child exceeded {timeout:.1f}s"},
                    }
                )
            try:
                received = receive.recv()
            except EOFError:
                received = {
                    "ok": False,
                    "error_type": "Gate2ChildExited",
                    "error": "Gate2 child closed its evidence pipe without a result",
                    "traceback": "",
                }
            if not isinstance(received, Mapping):
                received = {
                    "ok": False,
                    "error_type": "Gate2ChildProtocolError",
                    "error": "Gate2 child returned a non-object message",
                    "traceback": "",
                }
            message = received
        finally:
            receive.close()
            process.join(30.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
        if message.get("ok"):
            return _Gate2JsonResult(message["result"])
        return _Gate2JsonResult(
            {
                "suite_id": case_id,
                "passed": False,
                "failed_case_ids": ["gate2_process_failure"],
                "status_counts": {"process_failure": 1},
                "cases": [],
                "metadata": {
                    "error_type": message.get("error_type"),
                    "error": message.get("error"),
                    "traceback": message.get("traceback"),
                },
            }
        )


def _cohort_memberships(case_id: str) -> list[str]:
    frontend, raw_stem = case_id.split(":", 1)
    stem = raw_stem.removeprefix("tilelang_") if frontend == "cuda" else raw_stem
    manifest = json.loads(TASK_ROOT.joinpath("cohorts.json").read_text(encoding="utf-8"))
    return [
        name for name, members in manifest["cohorts"].items() if stem in members
    ]


def _preflight(case_id: str) -> tuple[Gate1Artifact, Any, Frontend, dict[str, Any]]:
    artifact = Gate1Artifact.from_json(CASE_BUILDERS[case_id]())
    prepared = prepare_registered_scrc_workload(case_id)
    frontend = Frontend(prepared.request.frontend)
    report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.requires_program_axes,
    )
    visibility = _visibility(artifact, frontend)
    row = {
        "case": case_id,
        "frontend": frontend.value,
        "typecheck": report.status.value,
        "diagnostics": [item.to_json() for item in report.diagnostics],
        "packet_root": visibility["packet_root"],
        "pi_visible_to_replay": visibility["pi_visible_to_replay"],
        "lift_agent_calls": 0,
        "cohort_memberships": _cohort_memberships(case_id),
    }
    return artifact, prepared, frontend, row


def _run_one(
    case_id: str,
    *,
    provider_name: str | None,
    effort: str,
    run_root: Path,
    gate2_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    run_dir = _next_run_dir(run_root, case_id)
    started = time.time()
    try:
        provider = get_provider(provider_name)
        artifact, prepared, frontend, preflight = _preflight(case_id)
        visibility = _visibility(artifact, frontend)

        run_dir.joinpath("source.py").write_text(
            prepared.request.source.text, encoding="utf-8"
        )
        _write_json(run_dir / "adapter.json", prepared.adapter.to_json())
        _write_json(run_dir / "artifact.json", artifact.to_json())
        _write_json(run_dir / "authoring.json", CASE_RECEIPTS[case_id]())
        _write_json(run_dir / "pi_visibility.json", visibility)
        _write_json(run_dir / "preflight.json", preflight)
        _write_json(
            run_dir / "request.json",
            {
                "case": case_id,
                "frontend": frontend.value,
                "provider": provider.name,
                "model": provider.model,
                "effort": effort,
                "lift_agent_calls": 0,
                "artifact_author": "primary Codex agent",
                "replay_backend": "codex",
                "replay_composer_mode": "off",
                "run_masked_j": False,
                "cohort_memberships": _cohort_memberships(case_id),
            },
        )

        gate1_started = time.time()
        recheck = recheck_prepared_workload_gate1(
            prepared,
            artifact,
            provider=provider,
            effort=effort,
            replay_backend="codex",
            replay_composer_mode="off",
            run_masked_j=False,
        )
        gate1_elapsed_ms = (time.time() - gate1_started) * 1000.0
        _write_json(run_dir / "typecheck.json", recheck.typecheck.to_json())
        if recheck.gate1 is not None:
            _write_json(run_dir / "gate1.json", recheck.gate1.to_json())

        records = recheck.replay_writer.records if recheck.replay_writer else ()
        _write_json(
            run_dir / "replay_records.json",
            [_replay_record(row) for row in records],
        )
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
            gate2 = _execute_gate2(
                case_id, recheck.artifact, recheck.gate1, gate2_semaphore
            )
            gate2_elapsed_ms = (time.time() - gate2_started) * 1000.0
            _write_json(run_dir / "gate2.json", gate2.to_json())

        gate2_status = None
        gate2_status_counts: dict[str, int] = {}
        if gate2 is not None:
            gate2_status = "gate2_pass" if gate2.passed else "gate2_fail"
            gate2_status_counts = dict(gate2.status_counts)
        summary = {
            "protocol": "intentir.experiment.manual-batch-j-only.v3",
            "case": case_id,
            "frontend": frontend.value,
            "typecheck": recheck.typecheck.status.value,
            "gate1": recheck.gate1.status.value if recheck.gate1 else None,
            "gate2": gate2_status,
            "gate2_passed": gate2.passed if gate2 is not None else None,
            "gate2_status_counts": gate2_status_counts,
            "pi_visible_to_replay": False,
            "lift_agent_calls": 0,
            "replay_provider_calls": sum(
                row.physical_provider_calls for row in records
            ),
            "gate1_elapsed_ms": gate1_elapsed_ms,
            "gate2_elapsed_ms": gate2_elapsed_ms,
            "total_elapsed_ms": (time.time() - started) * 1000.0,
            "cohort_memberships": _cohort_memberships(case_id),
            "run_dir": str(run_dir),
        }
        _write_json(run_dir / "timing.json", {
            "gate1_elapsed_ms": gate1_elapsed_ms,
            "gate2_elapsed_ms": gate2_elapsed_ms,
            "total_elapsed_ms": summary["total_elapsed_ms"],
        })
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        failure = {
            "protocol": "intentir.experiment.manual-batch-failure.v1",
            "case": case_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "lift_agent_calls": 0,
            "total_elapsed_ms": (time.time() - started) * 1000.0,
            "run_dir": str(run_dir),
        }
        _write_json(run_dir / "failure.json", failure)
        return failure


def _select_cases(args: argparse.Namespace) -> list[str]:
    if args.all_g1:
        return list(G1_CASE_IDS)
    if args.all_g2:
        return list(G2_CASE_IDS)
    if args.all_runnable_remaining:
        return list(G3_G7_CASE_IDS)
    if args.all_descriptor_remaining:
        return list(DESCRIPTOR_CASE_IDS)
    if args.all_authored:
        return list(CASE_IDS)
    if args.case:
        return list(dict.fromkeys(args.case))
    if args.preflight_only:
        return list(CASE_IDS)
    raise SystemExit(
        "select at least one --case or pass "
        "--all-g1/--all-g2/--all-runnable-remaining/"
        "--all-descriptor-remaining/--all-authored"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASE_BUILDERS))
    parser.add_argument("--all-g1", action="store_true")
    parser.add_argument("--all-g2", action="store_true")
    parser.add_argument("--all-runnable-remaining", action="store_true")
    parser.add_argument("--all-descriptor-remaining", action="store_true")
    parser.add_argument("--all-authored", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gate2-workers", type=int, default=1)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=TASK_ROOT / "runs",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.gate2_workers < 1:
        parser.error("worker counts must be positive")
    selected = _select_cases(args)

    preflight_rows = []
    for case_id in selected:
        _artifact, _prepared, _frontend, row = _preflight(case_id)
        preflight_rows.append(row)
    preflight_report = {
        "protocol": "intentir.experiment.manual-batch-preflight.v1",
        "case_count": len(preflight_rows),
        "all_typecheck_passed": all(row["typecheck"] == "pass" for row in preflight_rows),
        "all_j_only": all(not row["pi_visible_to_replay"] for row in preflight_rows),
        "lift_agent_calls": 0,
        "cases": preflight_rows,
    }
    if selected == list(G1_CASE_IDS):
        preflight_name = "preflight_g1.json"
    elif selected == list(G2_CASE_IDS):
        preflight_name = "preflight_g2.json"
    elif selected == list(G3_G7_CASE_IDS):
        preflight_name = "preflight_g3_g7_runnable.json"
    elif selected == list(DESCRIPTOR_CASE_IDS):
        preflight_name = "preflight_descriptor_remaining.json"
    elif selected == list(CASE_IDS):
        preflight_name = "preflight_authored.json"
    else:
        preflight_name = "preflight_latest.json"
    _write_json(TASK_ROOT / preflight_name, preflight_report)
    if args.preflight_only:
        print(json.dumps(preflight_report, sort_keys=True, ensure_ascii=False))
        return 0 if preflight_report["all_typecheck_passed"] and preflight_report["all_j_only"] else 2

    gate2_semaphore = threading.Semaphore(args.gate2_workers)
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as executor:
        futures = {
            executor.submit(
                _run_one,
                case_id,
                provider_name=args.provider,
                effort=args.effort,
                run_root=args.run_root,
                gate2_semaphore=gate2_semaphore,
            ): case_id
            for case_id in selected
        }
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)

    summaries.sort(key=lambda row: selected.index(str(row["case"])))
    batch = {
        "protocol": "intentir.experiment.manual-batch-summary.v1",
        "selected": selected,
        "workers": min(args.workers, len(selected)),
        "gate2_workers": args.gate2_workers,
        "lift_agent_calls": 0,
        "results": summaries,
        "accepted": [
            row["case"]
            for row in summaries
            if row.get("gate1") == "replay_closed" and row.get("gate2_passed") is True
        ],
    }
    batch["accepted_count"] = len(batch["accepted"])
    _write_json(TASK_ROOT / "batch_latest.json", batch)
    return 0 if batch["accepted_count"] == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
