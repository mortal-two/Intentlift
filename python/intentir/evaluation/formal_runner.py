"""Thin, provenance-closed runner for one frozen formal batch case.

This module does not interpret J, choose rules, or compose a program.  It only
forces the already-defined sequence ``Lift -> source-blind Replay -> Gate1``
and writes the exact request/response evidence before advancing state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from intentir.extraction.lifters.scrc import (
    SCRCLiftError,
    extract_scrc_artifact,
    prepare_scrc_lift,
)
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_text_format
from intentir.extraction.providers.client import (
    ProviderError,
    responses_request_body,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.orchestrate import (
    Gate1Config,
    Gate1Runner,
    canonical_artifact_digest,
)
from intentir.verify.gate1.prompt import (
    build_rewriter_prompt,
)
from intentir.verify.gate1.replay import (
    ReplayGeneration,
    ReplayGenerationError,
    ReplayWriter,
    ResponsesReplayWriter,
)

from .real_batch import (
    BatchManifest,
    compiler_pythonpath,
    load_batch_manifest,
    resolve_source_path,
    scrc_compiler_view,
    triton_compile_case,
)


RUN_SCHEMA = "intentir.formal-real-llm-case.v1"


class FormalRunError(RuntimeError):
    """The immutable formal call ledger cannot safely advance."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode())


def _record_history(receipt: dict[str, Any], status: str) -> None:
    receipt["status"] = status
    receipt.setdefault("history", []).append({"status": status, "at": _utc_now()})


def _persist_receipt(case_dir: Path, receipt: Mapping[str, Any]) -> None:
    _write_json(case_dir / "receipt.json", receipt)


def _relative(case_dir: Path, path: Path) -> str:
    return path.relative_to(case_dir).as_posix()


def _append_requested_call(
    case_dir: Path,
    receipt: dict[str, Any],
    *,
    kind: str,
    prompt_sha256: str,
    request_body: bytes,
    call_dir: Path,
) -> dict[str, Any]:
    if any(item["kind"] == kind for item in receipt["calls"]):
        raise FormalRunError(f"formal call {kind!r} is already present in the ledger")
    limit = int(receipt["call_budget"].get(kind, 0))
    if limit != 1:
        raise FormalRunError(f"formal call budget for {kind!r} is not exactly one")
    call_dir.mkdir(parents=True, exist_ok=False)
    request_path = call_dir / "request.json"
    _atomic_write(request_path, request_body)
    record = {
        "ordinal": len(receipt["calls"]) + 1,
        "kind": kind,
        "status": "requested",
        "requested_at": _utc_now(),
        "prompt_sha256": prompt_sha256,
        "request_sha256": _sha256(request_body),
        "request_path": _relative(case_dir, request_path),
    }
    receipt["calls"].append(record)
    _record_history(receipt, f"{kind}_requested")
    _persist_receipt(case_dir, receipt)
    return record


def _complete_call(
    case_dir: Path,
    receipt: dict[str, Any],
    record: dict[str, Any],
    *,
    response_text: str,
    raw_response: bytes,
    response_sha256: str,
    raw_response_sha256: str,
    response_id: str | None,
    response_model: str | None,
    usage: Mapping[str, int],
    attempts: int,
    elapsed_ms: float,
    call_dir: Path,
) -> None:
    if attempts != 1:
        raise FormalRunError("formal xhigh call used more than one HTTP attempt")
    response_path = call_dir / "response.txt"
    raw_path = call_dir / "response.raw.json"
    _write_text(response_path, response_text)
    if raw_response:
        _atomic_write(raw_path, raw_response)
    record.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "response_sha256": response_sha256 or _sha256(response_text.encode()),
            "raw_response_sha256": raw_response_sha256 or None,
            "response_path": _relative(case_dir, response_path),
            "raw_response_path": (
                _relative(case_dir, raw_path) if raw_response else None
            ),
            "response_id": response_id,
            "response_model": response_model,
            "usage": dict(usage),
            "attempts": attempts,
            "elapsed_ms": elapsed_ms,
        }
    )
    _persist_receipt(case_dir, receipt)


def _fail_call(
    case_dir: Path,
    receipt: dict[str, Any],
    record: dict[str, Any],
    error: BaseException,
    *,
    response_text: str = "",
    raw_response: bytes = b"",
    response_sha256: str = "",
    raw_response_sha256: str = "",
    response_id: str | None = None,
    response_model: str | None = None,
    usage: Mapping[str, int] | None = None,
    attempts: int | None = None,
    elapsed_ms: float | None = None,
) -> None:
    call_dir = case_dir / Path(record["request_path"]).parent
    if response_text:
        _write_text(call_dir / "response.txt", response_text)
    if raw_response:
        _atomic_write(call_dir / "response.raw.json", raw_response)
    record.update(
        {
            "status": "failed",
            "failed_at": _utc_now(),
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    )
    if response_text:
        record.update(
            {
                "response_path": _relative(case_dir, call_dir / "response.txt"),
                "response_sha256": response_sha256
                or _sha256(response_text.encode()),
                "response_id": response_id,
                "response_model": response_model,
                "usage": dict(usage or {}),
            }
        )
    if raw_response:
        record.update(
            {
                "raw_response_path": _relative(
                    case_dir, call_dir / "response.raw.json"
                ),
                "raw_response_sha256": raw_response_sha256
                or _sha256(raw_response),
            }
        )
    if attempts is not None:
        record["attempts"] = attempts
    if elapsed_ms is not None:
        record["elapsed_ms"] = elapsed_ms
    if isinstance(error, ProviderError):
        record.update(
            {
                "request_sha256": error.request_sha256,
                "attempts": error.attempts,
                "elapsed_ms": error.elapsed_ms,
            }
        )
    _record_history(receipt, f"{record['kind']}_failed")
    _persist_receipt(case_dir, receipt)


class LedgerReplayWriter(ReplayWriter):
    """Replay writer decorator that records calls without changing prompts."""

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        effort: str,
        case_dir: Path,
        receipt: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.effort = effort
        self.case_dir = case_dir
        self.receipt = receipt
        self.inner = ResponsesReplayWriter(provider=provider, effort=effort)

    def generate(self, artifact, *, masked: bool = False) -> ReplayGeneration:
        mode = "masked_j" if masked else "normal_replay"
        instructions, user_input = build_rewriter_prompt(
            artifact,
            masked=masked,
            frontend_context=self.inner.frontend_context,
        )
        prompt_sha = _sha256((instructions + "\0" + user_input).encode())
        request_body = responses_request_body(
            instructions,
            user_input,
            provider=self.provider,
            effort=self.effort,
        )
        call_dir = self.case_dir / "calls" / f"{len(self.receipt['calls']) + 1:02d}_{mode}"
        record = _append_requested_call(
            self.case_dir,
            self.receipt,
            kind=mode,
            prompt_sha256=prompt_sha,
            request_body=request_body,
            call_dir=call_dir,
        )
        try:
            generation = self.inner.generate(artifact, masked=masked)
            if generation.prompt_digest != prompt_sha:
                raise FormalRunError("Replay prompt changed between ledger and provider call")
            if generation.request_sha256 != record["request_sha256"]:
                raise FormalRunError("Replay request bytes differ from the ledger")
            _complete_call(
                self.case_dir,
                self.receipt,
                record,
                response_text=generation.response_text,
                raw_response=generation.raw_response,
                response_sha256=generation.response_sha256,
                raw_response_sha256=generation.raw_response_sha256,
                response_id=generation.response_id,
                response_model=generation.response_model or generation.model,
                usage=generation.usage,
                attempts=generation.attempts,
                elapsed_ms=generation.elapsed_ms,
                call_dir=call_dir,
            )
            code_path = call_dir / "replay.py"
            _write_text(code_path, generation.code)
            record["replay_code_path"] = _relative(self.case_dir, code_path)
            record["replay_code_sha256"] = _sha256(generation.code.encode())
            record["packet_sha256"] = generation.packet_digest
            _persist_receipt(self.case_dir, self.receipt)
            return generation
        except ReplayGenerationError as exc:
            attempt = exc.attempt
            _fail_call(
                self.case_dir,
                self.receipt,
                record,
                exc,
                response_text=attempt.response_text,
                raw_response=attempt.raw_response,
                response_sha256=attempt.response_sha256,
                raw_response_sha256=attempt.raw_response_sha256,
                response_id=attempt.response_id,
                response_model=attempt.response_model or attempt.model,
                usage=attempt.usage,
                attempts=attempt.attempts,
                elapsed_ms=attempt.elapsed_ms,
            )
            raise
        except BaseException as exc:
            _fail_call(self.case_dir, self.receipt, record, exc)
            raise


def _initial_receipt(
    manifest: BatchManifest,
    manifest_path: Path,
    case: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    pin = case["source"]
    receipt: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "batch_id": manifest.data["batch_id"],
        "case_id": case["case_id"],
        "manifest_digest": manifest.digest,
        "manifest_file_sha256": _sha256(manifest_path.read_bytes()),
        "source": {
            "repository": pin["repository"],
            "path": pin["path"],
            "symbol": pin["symbol"],
            "frontend": pin["frontend"],
            "file_sha256": pin["file_sha256"],
            "symbol_sha256": pin["symbol_sha256"],
            "resolved_path_sha256": _sha256(str(source_path).encode()),
        },
        "provider": dict(manifest.data["provider"]),
        "call_budget": {
            "lift": manifest.data["call_budget"]["lift_per_case"],
            "normal_replay": manifest.data["call_budget"]["normal_replay_per_case"],
            "masked_j": manifest.data["call_budget"]["masked_j_per_closed_case"],
        },
        "calls": [],
        "artifacts": {},
        "history": [],
    }
    _record_history(receipt, "prepared")
    return receipt


def _validate_preflight(manifest: BatchManifest, preflight_path: Path) -> None:
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"cannot read frozen preflight: {exc}") from exc
    if preflight.get("manifest_digest") != manifest.digest:
        raise FormalRunError("preflight is not bound to the active manifest")
    if preflight.get("online_llm_calls") != 0:
        raise FormalRunError("preflight already reports online calls")
    if preflight.get("summary", {}).get("overall_ready") is not True:
        raise FormalRunError("formal preflight is not 28/28 ready")


def run_formal_gate1_case(
    manifest_path: str | os.PathLike[str],
    case_id: str,
    *,
    output_root: str | os.PathLike[str],
    preflight_path: str | os.PathLike[str],
    compiler_timeout: float = 300.0,
) -> dict[str, Any]:
    """Spend one Lift/Replay budget for a frozen Triton case and run Gate1."""

    manifest_path = Path(manifest_path).resolve()
    manifest = load_batch_manifest(
        manifest_path, verify_sources=True, verify_repositories=True
    )
    _validate_preflight(manifest, Path(preflight_path).resolve())
    case = manifest.case(case_id)
    if case["source"]["frontend"] != "triton":
        raise FormalRunError(
            "formal runner currently requires the Triton Gate1 orchestrator; "
            "CuTe calls remain forbidden until its replay arm is wired"
        )
    case_dir = Path(output_root).resolve() / case_id
    receipt_path = case_dir / "receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") == "gate1_closed":
            return existing
        raise FormalRunError(
            "an incomplete formal ledger already exists; automatic retry is forbidden"
        )
    case_dir.mkdir(parents=True, exist_ok=False)
    source_path = resolve_source_path(manifest, case)
    source = source_path.read_text(encoding="utf-8")
    receipt = _initial_receipt(manifest, manifest_path, case, source_path)
    _persist_receipt(case_dir, receipt)

    provider_spec = manifest.data["provider"]
    provider = get_provider(provider_spec["name"])
    effort = str(provider_spec["reasoning_effort"])
    if provider.model != provider_spec["model"]:
        raise FormalRunError("configured provider model differs from the frozen manifest")

    compiler_view = scrc_compiler_view(case)
    lift_request = prepare_scrc_lift(
        source,
        compiler_view=compiler_view,
        source_symbol=case["source"]["symbol"],
    )
    lift_format = scrc_lift_core_text_format()
    lift_body = responses_request_body(
        lift_request.instructions,
        lift_request.user_input,
        provider=provider,
        effort=effort,
        text_format=lift_format,
    )
    lift_dir = case_dir / "calls" / "01_lift"
    lift_record = _append_requested_call(
        case_dir,
        receipt,
        kind="lift",
        prompt_sha256=lift_request.prompt_sha256,
        request_body=lift_body,
        call_dir=lift_dir,
    )
    _write_text(lift_dir / "instructions.txt", lift_request.instructions)
    _write_text(lift_dir / "input.json", lift_request.user_input)
    try:
        extraction = extract_scrc_artifact(
            source,
            provider=provider,
            compiler_view=compiler_view,
            effort=effort,
            source_symbol=case["source"]["symbol"],
        )
        provenance = extraction.provenance
        if provenance.prompt_sha256 != lift_request.prompt_sha256:
            raise FormalRunError("Lift prompt changed between ledger and provider call")
        if provenance.request_sha256 != lift_record["request_sha256"]:
            raise FormalRunError("Lift request bytes differ from the ledger")
        _complete_call(
            case_dir,
            receipt,
            lift_record,
            response_text=provenance.response_text,
            raw_response=provenance.raw_response,
            response_sha256=provenance.response_sha256,
            raw_response_sha256=provenance.raw_response_sha256,
            response_id=provenance.response_id,
            response_model=provenance.response_model or provenance.model,
            usage=provenance.usage,
            attempts=provenance.attempts,
            elapsed_ms=provenance.elapsed_ms,
            call_dir=lift_dir,
        )
    except SCRCLiftError as exc:
        attempt = exc.attempts[0]
        _fail_call(
            case_dir,
            receipt,
            lift_record,
            exc,
            response_text=attempt.response_text,
            raw_response=attempt.raw_response,
            response_sha256=attempt.response_sha256,
            raw_response_sha256=attempt.raw_response_sha256,
            response_id=attempt.response_id,
            response_model=attempt.response_model or attempt.model,
            usage=attempt.usage,
            attempts=attempt.attempts,
            elapsed_ms=attempt.elapsed_ms,
        )
        raise
    except BaseException as exc:
        _fail_call(case_dir, receipt, lift_record, exc)
        raise

    artifact = extraction.artifact
    if artifact.interface.frontend.value != case["source"]["frontend"]:
        raise FormalRunError("Lift returned the wrong public frontend")
    public_artifact = artifact.to_json()
    public_artifact.pop("source", None)
    artifact_path = case_dir / "lift" / "artifact.json"
    binding_path = case_dir / "lift" / "privileged_binding.json"
    _write_json(artifact_path, public_artifact)
    _write_json(
        binding_path,
        {
            "source_function": extraction.harness_binding.source_function,
            "replay_function": extraction.harness_binding.replay_function,
            "source_to_opaque": [
                list(item) for item in extraction.harness_binding.source_to_opaque
            ],
        },
    )
    receipt["artifacts"] = {
        "artifact_path": _relative(case_dir, artifact_path),
        "artifact_sha256": _sha256(_canonical_bytes(public_artifact)),
        "canonical_gate1_sha256": canonical_artifact_digest(artifact),
        "j_sha256": _sha256(_canonical_bytes(artifact.j.to_json())),
        "c_sha256": _sha256(_canonical_bytes(artifact.schedule.to_json())),
        "pi_sha256": _sha256(_canonical_bytes(artifact.pi.to_json())),
        "interface_sha256": _sha256(_canonical_bytes(artifact.interface.to_json())),
        "privileged_binding_path": _relative(case_dir, binding_path),
    }
    _record_history(receipt, "lift_closed")
    _persist_receipt(case_dir, receipt)

    writer = LedgerReplayWriter(
        provider=provider,
        effort=effort,
        case_dir=case_dir,
        receipt=receipt,
    )
    compiler = TritonCompiler(
        timeout=compiler_timeout,
        cache_dir=case_dir / "compiler_cache",
        env={"PYTHONPATH": compiler_pythonpath(manifest)},
    )
    runner = Gate1Runner(
        compiler=compiler,
        writer=writer,
        config=Gate1Config(observer_level=2, run_masked_j=True),
    )
    result = runner.run(
        artifact,
        source=KernelSource.from_file(
            source_path,
            case["source"]["symbol"],
            trusted=True,
        ),
        source_case=triton_compile_case(case),
    )
    gate1_path = case_dir / "gate1" / "receipt.json"
    _write_json(gate1_path, result.to_json())
    receipt["gate1"] = {
        "status": result.status.value,
        "passed": result.passed,
        "protocol": result.protocol,
        "artifact_sha256": result.artifact_digest,
        "receipt_path": _relative(case_dir, gate1_path),
        "receipt_sha256": _sha256(_canonical_bytes(result.to_json())),
        "normal_calls": result.normal_calls,
        "masked_j_calls": result.masked_j_calls,
    }
    _record_history(receipt, "gate1_closed" if result.passed else "gate1_failed")
    _persist_receipt(case_dir, receipt)
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--compiler-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    receipt = run_formal_gate1_case(
        args.manifest,
        args.case_id,
        output_root=args.output_root,
        preflight_path=args.preflight,
        compiler_timeout=args.compiler_timeout,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "gate1_closed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "FormalRunError",
    "LedgerReplayWriter",
    "RUN_SCHEMA",
    "run_formal_gate1_case",
]
