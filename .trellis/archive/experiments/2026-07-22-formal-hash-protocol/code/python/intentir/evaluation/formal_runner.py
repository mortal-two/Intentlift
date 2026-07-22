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
from intentir.evaluation.lift_wire import LiftWireError, verify_lift_wire_lock
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.cute_compiler import CuteCompiler, CuteKernelSource
from intentir.verify.gate1.cute_orchestrate import CuteGate1Runner
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
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V4,
    SCHEMA_V5,
    SCHEMA_V6,
    SCHEMA_V7,
    V2_BATCH_ID,
    V2_MANIFEST_DIGEST,
    V2_MANIFEST_FILE_SHA256,
    V3_BATCH_ID,
    V3_MANIFEST_DIGEST,
    V3_MANIFEST_FILE_SHA256,
    V4_BATCH_ID,
    V4_MANIFEST_DIGEST,
    V4_MANIFEST_FILE_SHA256,
    V5_BATCH_ID,
    V5_MANIFEST_DIGEST,
    V5_MANIFEST_FILE_SHA256,
    V6_BATCH_ID,
    V6_MANIFEST_DIGEST,
    V6_MANIFEST_FILE_SHA256,
    compiler_pythonpath,
    cute_compile_case,
    cute_entry_symbol,
    load_batch_manifest,
    resolve_source_path,
    scrc_compiler_view,
    triton_compile_case,
)
from .formal_preflight import verify_formal_preflight


RUN_SCHEMA = "intentir.formal-real-llm-case.v1"
RUN_SCHEMA_V2 = "intentir.formal-real-llm-case.v2"
RUN_SCHEMA_V3 = "intentir.formal-real-llm-case.v3"
RUN_SCHEMA_V4 = "intentir.formal-real-llm-case.v4"
RUN_SCHEMA_V5 = "intentir.formal-real-llm-case.v5"
RUN_SCHEMA_V6 = "intentir.formal-real-llm-case.v6"
RUN_SCHEMA_V7 = "intentir.formal-real-llm-case.v7"
_V2_ATTEMPT_CLASSES = frozenset(
    {"protocol_v2_first_attempt", "protocol_v2_recovery"}
)
_V3_ATTEMPT_CLASSES = frozenset(
    {"protocol_v3_first_attempt", "protocol_v3_recovery"}
)
_V4_ATTEMPT_CLASSES = frozenset(
    {"protocol_v4_first_attempt", "protocol_v4_recovery"}
)
_V5_ATTEMPT_CLASSES = frozenset(
    {"protocol_v5_first_attempt", "protocol_v5_recovery"}
)
_V6_ATTEMPT_CLASSES = frozenset(
    {"protocol_v6_first_attempt", "protocol_v6_recovery"}
)
_V7_ATTEMPT_CLASSES = frozenset(
    {"protocol_v7_first_attempt", "protocol_v7_recovery"}
)
_ATTEMPT_CLASSES_BY_SCHEMA = {
    SCHEMA_V2: _V2_ATTEMPT_CLASSES,
    SCHEMA_V3: _V3_ATTEMPT_CLASSES,
    SCHEMA_V4: _V4_ATTEMPT_CLASSES,
    SCHEMA_V5: _V5_ATTEMPT_CLASSES,
    SCHEMA_V6: _V6_ATTEMPT_CLASSES,
    SCHEMA_V7: _V7_ATTEMPT_CLASSES,
}
_RUN_SCHEMA_BY_MANIFEST = {
    SCHEMA_V1: RUN_SCHEMA,
    SCHEMA_V2: RUN_SCHEMA_V2,
    SCHEMA_V3: RUN_SCHEMA_V3,
    SCHEMA_V4: RUN_SCHEMA_V4,
    SCHEMA_V5: RUN_SCHEMA_V5,
    SCHEMA_V6: RUN_SCHEMA_V6,
    SCHEMA_V7: RUN_SCHEMA_V7,
}

_V4_RECOVERY_RECEIPT_SHA256 = (
    "98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2"
)
_V4_PREFLIGHT_SHA256 = (
    "74862a1402776d68988f3e704b2ceb22d61e89a5849e6a8c1da6c278e585fa30"
)
_V4_INVALID_RESPONSE_SHA256 = (
    "289aa3b37fb04c198941f31f76e31a4d05b118049dcf149e4593e8bde8e7db0a"
)
_V3_RECOVERY_RECEIPT_SHA256 = (
    "ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1"
)
_V5_RECOVERY_RECEIPT_SHA256 = (
    "5a286725e22f1fadc83771eb460a26e823e0c9675606bde311412563f2921b71"
)
_V5_PREFLIGHT_SHA256 = (
    "f0c93e6230914a9f750173ef6853315b9a1df9c212c4592e0a258a6f48222b6e"
)
_V5_ARTIFACT_SHA256 = (
    "a3ec791ee55ffbac51aac2dc68fe06d7cb45a85a3a2b497e2bbfb77b931ff68e"
)
_V5_GATE1_RECEIPT_SHA256 = (
    "e77401db2db33db8b4084f0cc8aa3eb5380d896c89fcd321eeb32c89ef4e3583"
)
_V6_RECOVERY_RECEIPT_SHA256 = (
    "21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d"
)
_V6_PREFLIGHT_SHA256 = (
    "ffda048dfdf00738359e4fe335d9b6d188bcd7590fdcc9297f7cf9e2052fec52"
)
_V6_REQUEST_SHA256 = (
    "cc44b107c1bad9ffa89f9489de539dc078bd81f167d297ec149f568f30cf3d9f"
)
_V6_ENVELOPE_RESPONSE_SHA256 = (
    "4ec0e7f4b8ef75ac0e0794ece900e8cb00e7e7575d2823a0aec74ce59f7d01fb"
)
_V6_RAW_RESPONSE_SHA256 = (
    "60c6242c419c79bbd2215842b8e88af8d93ef4fa0588915601fc669ffa1f0700"
)


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
            record["replay_function"] = generation.function_name
            record["replay_entry_function"] = generation.entry_function_name
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
    preflight_path: Path,
    preflight: Mapping[str, Any],
    lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pin = case["source"]
    receipt: dict[str, Any] = {
        "schema": _RUN_SCHEMA_BY_MANIFEST[str(manifest.data["schema"])],
        "batch_id": manifest.data["batch_id"],
        "case_id": case["case_id"],
        "manifest_digest": manifest.digest,
        "manifest_file_sha256": _sha256(manifest_path.read_bytes()),
        "preflight": {
            "schema": preflight.get("schema"),
            "sha256": _sha256(preflight_path.read_bytes()),
        },
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
    if lineage is not None:
        receipt["lineage"] = dict(lineage)
    _record_history(receipt, "prepared")
    return receipt


def _validate_preflight(
    manifest: BatchManifest, preflight_path: Path, manifest_path: Path
) -> Mapping[str, Any]:
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"cannot read frozen preflight: {exc}") from exc
    if "lift_wire" in manifest.data:
        try:
            verify_formal_preflight(preflight, manifest, manifest_path)
        except (ValueError, OSError) as exc:
            raise FormalRunError(str(exc)) from exc
    else:
        if preflight.get("manifest_digest") != manifest.digest:
            raise FormalRunError("preflight is not bound to the active manifest")
        if preflight.get("online_llm_calls") != 0:
            raise FormalRunError("preflight already reports online calls")
        if preflight.get("summary", {}).get("overall_ready") is not True:
            raise FormalRunError("formal preflight is not 28/28 ready")
    return preflight


def _lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str | None,
    predecessor_receipt: str | os.PathLike[str] | None,
) -> Mapping[str, Any] | None:
    manifest_schema = str(manifest.data["schema"])
    if manifest_schema == SCHEMA_V1:
        if attempt_class is not None or predecessor_receipt is not None:
            raise FormalRunError("v1 formal runs do not accept lineage metadata")
        return None
    allowed = _ATTEMPT_CLASSES_BY_SCHEMA.get(manifest_schema)
    if allowed is None or attempt_class not in allowed:
        raise FormalRunError(
            f"{manifest_schema} requires its explicit attempt classification"
        )
    first_attempt, recovery = sorted(allowed)
    if attempt_class == first_attempt:
        if predecessor_receipt is not None:
            raise FormalRunError("a first attempt may not name a predecessor")
        return {"attempt_class": attempt_class}
    assert attempt_class == recovery
    if predecessor_receipt is None:
        raise FormalRunError("protocol recovery requires its predecessor receipt")
    path = Path(predecessor_receipt).resolve()
    try:
        payload = path.read_bytes()
        predecessor = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"cannot read recovery predecessor: {exc}") from exc
    if manifest_schema == SCHEMA_V4:
        return _v4_recovery_lineage(
            manifest,
            case_id,
            attempt_class,
            predecessor,
            payload,
        )
    if manifest_schema == SCHEMA_V5:
        return _v5_recovery_lineage(
            manifest,
            case_id,
            attempt_class,
            predecessor,
            payload,
        )
    if manifest_schema == SCHEMA_V6:
        return _v6_recovery_lineage(
            manifest,
            case_id,
            attempt_class,
            predecessor,
            payload,
        )
    if manifest_schema == SCHEMA_V7:
        return _v7_recovery_lineage(
            manifest,
            case_id,
            attempt_class,
            predecessor,
            payload,
        )
    if (
        predecessor.get("case_id") != case_id
        or predecessor.get("status") != "lift_failed"
        or predecessor.get("batch_id") == manifest.data["batch_id"]
    ):
        raise FormalRunError("recovery predecessor is not the prior failed case epoch")
    calls = predecessor.get("calls")
    if (
        not isinstance(calls, list)
        or len(calls) != 1
        or calls[0].get("kind") != "lift"
        or calls[0].get("status") != "failed"
        or calls[0].get("attempts") != 1
        or predecessor.get("gate1") is not None
    ):
        raise FormalRunError("recovery predecessor lacks one auditable failed call")
    case = manifest.case(case_id)
    source = predecessor.get("source")
    expected_source = case["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in (
            "repository", "path", "symbol", "frontend", "file_sha256",
            "symbol_sha256",
        )
    ):
        raise FormalRunError("recovery predecessor source identity differs")
    if predecessor.get("provider") != manifest.data["provider"]:
        raise FormalRunError("recovery predecessor provider identity differs")
    if manifest_schema == SCHEMA_V3:
        predecessor_lineage = predecessor.get("lineage")
        if (
            predecessor.get("schema") != RUN_SCHEMA_V2
            or predecessor.get("batch_id") != V2_BATCH_ID
            or predecessor.get("manifest_digest") != V2_MANIFEST_DIGEST
            or predecessor.get("manifest_file_sha256")
            != V2_MANIFEST_FILE_SHA256
            or not isinstance(predecessor_lineage, dict)
            or predecessor_lineage.get("attempt_class")
            != "protocol_v2_recovery"
        ):
            raise FormalRunError(
                "protocol-v3 recovery requires the frozen protocol-v2 recovery"
            )
    return {
        "attempt_class": attempt_class,
        "predecessor": {
            "schema": predecessor.get("schema"),
            "batch_id": predecessor.get("batch_id"),
            "case_id": predecessor["case_id"],
            "status": predecessor["status"],
            "attempt_class": (
                predecessor.get("lineage", {}).get("attempt_class")
                if isinstance(predecessor.get("lineage"), dict)
                else None
            ),
            "manifest_digest": predecessor.get("manifest_digest"),
            "manifest_file_sha256": predecessor.get("manifest_file_sha256"),
            "receipt_sha256": _sha256(payload),
        },
    }


def _v4_recovery_lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str,
    predecessor: Mapping[str, Any],
    payload: bytes,
) -> Mapping[str, Any]:
    """Bind v4 recovery to the frozen v3 strict Gate1 no-closure."""

    if attempt_class != "protocol_v4_recovery":
        raise FormalRunError("protocol-v4 recovery classification is invalid")
    if (
        predecessor.get("case_id") != case_id
        or predecessor.get("status") != "gate1_failed"
        or predecessor.get("schema") != RUN_SCHEMA_V3
        or predecessor.get("batch_id") != V3_BATCH_ID
        or predecessor.get("manifest_digest") != V3_MANIFEST_DIGEST
        or predecessor.get("manifest_file_sha256")
        != V3_MANIFEST_FILE_SHA256
    ):
        raise FormalRunError(
            "protocol-v4 recovery requires the frozen protocol-v3 Gate1 failure"
        )
    calls = predecessor.get("calls")
    if (
        not isinstance(calls, list)
        or [
            (item.get("kind"), item.get("status"), item.get("attempts"))
            for item in calls
        ]
        != [
            ("lift", "completed", 1),
            ("normal_replay", "completed", 1),
        ]
    ):
        raise FormalRunError(
            "protocol-v4 predecessor lacks one Lift and one normal Replay"
        )
    gate1 = predecessor.get("gate1")
    if (
        not isinstance(gate1, dict)
        or gate1.get("status") != "no_closure"
        or gate1.get("passed") is not False
        or gate1.get("normal_calls") != 1
        or gate1.get("masked_j_calls") != 0
    ):
        raise FormalRunError(
            "protocol-v4 predecessor is not the frozen strict Gate1 no-closure"
        )
    predecessor_lineage = predecessor.get("lineage")
    if (
        not isinstance(predecessor_lineage, dict)
        or predecessor_lineage.get("attempt_class")
        != "protocol_v3_recovery"
    ):
        raise FormalRunError(
            "protocol-v4 predecessor lacks protocol-v3 recovery lineage"
        )
    case = manifest.case(case_id)
    source = predecessor.get("source")
    expected_source = case["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in (
            "repository", "path", "symbol", "frontend", "file_sha256",
            "symbol_sha256",
        )
    ):
        raise FormalRunError("protocol-v4 predecessor source identity differs")
    if predecessor.get("provider") != manifest.data["provider"]:
        raise FormalRunError("protocol-v4 predecessor provider identity differs")
    return {
        "attempt_class": attempt_class,
        "predecessor": {
            "schema": predecessor["schema"],
            "batch_id": predecessor["batch_id"],
            "case_id": predecessor["case_id"],
            "status": predecessor["status"],
            "attempt_class": predecessor_lineage["attempt_class"],
            "manifest_digest": predecessor["manifest_digest"],
            "manifest_file_sha256": predecessor["manifest_file_sha256"],
            "receipt_sha256": _sha256(payload),
        },
    }


def _v5_recovery_lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str,
    predecessor: Mapping[str, Any],
    payload: bytes,
) -> Mapping[str, Any]:
    """Bind v5 recovery to the exact frozen v4 pi-endpoint rejection."""

    if attempt_class != "protocol_v5_recovery":
        raise FormalRunError("protocol-v5 recovery classification is invalid")
    if _sha256(payload) != _V4_RECOVERY_RECEIPT_SHA256:
        raise FormalRunError(
            "protocol-v5 recovery requires the exact frozen protocol-v4 receipt"
        )
    if (
        predecessor.get("case_id") != case_id
        or predecessor.get("status") != "lift_failed"
        or predecessor.get("schema") != RUN_SCHEMA_V4
        or predecessor.get("batch_id") != V4_BATCH_ID
        or predecessor.get("manifest_digest") != V4_MANIFEST_DIGEST
        or predecessor.get("manifest_file_sha256")
        != V4_MANIFEST_FILE_SHA256
    ):
        raise FormalRunError(
            "protocol-v5 recovery requires the frozen protocol-v4 Lift failure"
        )
    calls = predecessor.get("calls")
    if (
        not isinstance(calls, list)
        or len(calls) != 1
        or calls[0].get("kind") != "lift"
        or calls[0].get("status") != "failed"
        or calls[0].get("attempts") != 1
        or calls[0].get("response_sha256") != _V4_INVALID_RESPONSE_SHA256
        or "pi route instance is unknown" not in str(calls[0].get("diagnostic"))
        or predecessor.get("artifacts") != {}
        or predecessor.get("gate1") is not None
    ):
        raise FormalRunError(
            "protocol-v5 predecessor is not the one-call pi endpoint rejection"
        )
    preflight = predecessor.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "intentir.real-llm-batch-preflight.v4"
        or preflight.get("sha256") != _V4_PREFLIGHT_SHA256
    ):
        raise FormalRunError(
            "protocol-v5 predecessor lacks the frozen protocol-v4 preflight"
        )
    predecessor_lineage = predecessor.get("lineage")
    previous = (
        predecessor_lineage.get("predecessor")
        if isinstance(predecessor_lineage, dict)
        else None
    )
    if (
        not isinstance(predecessor_lineage, dict)
        or predecessor_lineage.get("attempt_class")
        != "protocol_v4_recovery"
        or not isinstance(previous, dict)
        or previous.get("receipt_sha256") != _V3_RECOVERY_RECEIPT_SHA256
    ):
        raise FormalRunError(
            "protocol-v5 predecessor lacks exact protocol-v4 recovery lineage"
        )
    case = manifest.case(case_id)
    source = predecessor.get("source")
    expected_source = case["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in (
            "repository", "path", "symbol", "frontend", "file_sha256",
            "symbol_sha256",
        )
    ):
        raise FormalRunError("protocol-v5 predecessor source identity differs")
    if predecessor.get("provider") != manifest.data["provider"]:
        raise FormalRunError("protocol-v5 predecessor provider identity differs")
    return {
        "attempt_class": attempt_class,
        "predecessor": {
            "schema": predecessor["schema"],
            "batch_id": predecessor["batch_id"],
            "case_id": predecessor["case_id"],
            "status": predecessor["status"],
            "attempt_class": predecessor_lineage["attempt_class"],
            "manifest_digest": predecessor["manifest_digest"],
            "manifest_file_sha256": predecessor["manifest_file_sha256"],
            "receipt_sha256": _sha256(payload),
        },
    }


def _v6_recovery_lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str,
    predecessor: Mapping[str, Any],
    payload: bytes,
) -> Mapping[str, Any]:
    """Bind v6 recovery to the exact frozen v5 address-topology failure."""

    if attempt_class != "protocol_v6_recovery":
        raise FormalRunError("protocol-v6 recovery classification is invalid")
    if _sha256(payload) != _V5_RECOVERY_RECEIPT_SHA256:
        raise FormalRunError(
            "protocol-v6 recovery requires the exact frozen protocol-v5 receipt"
        )
    if (
        predecessor.get("case_id") != case_id
        or predecessor.get("status") != "gate1_failed"
        or predecessor.get("schema") != RUN_SCHEMA_V5
        or predecessor.get("batch_id") != V5_BATCH_ID
        or predecessor.get("manifest_digest") != V5_MANIFEST_DIGEST
        or predecessor.get("manifest_file_sha256")
        != V5_MANIFEST_FILE_SHA256
    ):
        raise FormalRunError(
            "protocol-v6 recovery requires the frozen protocol-v5 Gate1 failure"
        )
    calls = predecessor.get("calls")
    if (
        not isinstance(calls, list)
        or [
            (item.get("kind"), item.get("status"), item.get("attempts"))
            for item in calls
        ]
        != [
            ("lift", "completed", 1),
            ("normal_replay", "completed", 1),
        ]
    ):
        raise FormalRunError(
            "protocol-v6 predecessor lacks one Lift and one normal Replay"
        )
    gate1 = predecessor.get("gate1")
    if (
        not isinstance(gate1, dict)
        or gate1.get("status") != "no_closure"
        or gate1.get("passed") is not False
        or gate1.get("normal_calls") != 1
        or gate1.get("masked_j_calls") != 0
        or gate1.get("artifact_sha256") != _V5_ARTIFACT_SHA256
        or gate1.get("receipt_sha256") != _V5_GATE1_RECEIPT_SHA256
    ):
        raise FormalRunError(
            "protocol-v6 predecessor is not the frozen strict address no-closure"
        )
    preflight = predecessor.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "intentir.real-llm-batch-preflight.v5"
        or preflight.get("sha256") != _V5_PREFLIGHT_SHA256
    ):
        raise FormalRunError(
            "protocol-v6 predecessor lacks the frozen protocol-v5 preflight"
        )
    predecessor_lineage = predecessor.get("lineage")
    previous = (
        predecessor_lineage.get("predecessor")
        if isinstance(predecessor_lineage, dict)
        else None
    )
    if (
        not isinstance(predecessor_lineage, dict)
        or predecessor_lineage.get("attempt_class")
        != "protocol_v5_recovery"
        or not isinstance(previous, dict)
        or previous.get("receipt_sha256") != _V4_RECOVERY_RECEIPT_SHA256
    ):
        raise FormalRunError(
            "protocol-v6 predecessor lacks exact protocol-v5 recovery lineage"
        )
    case = manifest.case(case_id)
    source = predecessor.get("source")
    expected_source = case["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in (
            "repository", "path", "symbol", "frontend", "file_sha256",
            "symbol_sha256",
        )
    ):
        raise FormalRunError("protocol-v6 predecessor source identity differs")
    if predecessor.get("provider") != manifest.data["provider"]:
        raise FormalRunError("protocol-v6 predecessor provider identity differs")
    return {
        "attempt_class": attempt_class,
        "predecessor": {
            "schema": predecessor["schema"],
            "batch_id": predecessor["batch_id"],
            "case_id": predecessor["case_id"],
            "status": predecessor["status"],
            "attempt_class": predecessor_lineage["attempt_class"],
            "manifest_digest": predecessor["manifest_digest"],
            "manifest_file_sha256": predecessor["manifest_file_sha256"],
            "receipt_sha256": _sha256(payload),
        },
    }


def _v7_recovery_lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str,
    predecessor: Mapping[str, Any],
    payload: bytes,
) -> Mapping[str, Any]:
    """Bind v7 recovery to the exact frozen v6 envelope rejection."""

    if attempt_class != "protocol_v7_recovery":
        raise FormalRunError("protocol-v7 recovery classification is invalid")
    if _sha256(payload) != _V6_RECOVERY_RECEIPT_SHA256:
        raise FormalRunError(
            "protocol-v7 recovery requires the exact frozen protocol-v6 receipt"
        )
    if (
        predecessor.get("case_id") != case_id
        or predecessor.get("status") != "lift_failed"
        or predecessor.get("schema") != RUN_SCHEMA_V6
        or predecessor.get("batch_id") != V6_BATCH_ID
        or predecessor.get("manifest_digest") != V6_MANIFEST_DIGEST
        or predecessor.get("manifest_file_sha256")
        != V6_MANIFEST_FILE_SHA256
    ):
        raise FormalRunError(
            "protocol-v7 recovery requires the frozen protocol-v6 Lift failure"
        )
    calls = predecessor.get("calls")
    if (
        not isinstance(calls, list)
        or len(calls) != 1
        or calls[0].get("kind") != "lift"
        or calls[0].get("status") != "failed"
        or calls[0].get("attempts") != 1
        or calls[0].get("request_sha256") != _V6_REQUEST_SHA256
        or calls[0].get("response_sha256")
        != _V6_ENVELOPE_RESPONSE_SHA256
        or calls[0].get("raw_response_sha256") != _V6_RAW_RESPONSE_SHA256
        or "Gate1Artifact is missing required key(s)" not in str(
            calls[0].get("diagnostic")
        )
        or predecessor.get("artifacts") != {}
        or predecessor.get("gate1") is not None
    ):
        raise FormalRunError(
            "protocol-v7 predecessor is not the one-call envelope rejection"
        )
    preflight = predecessor.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "intentir.real-llm-batch-preflight.v6"
        or preflight.get("sha256") != _V6_PREFLIGHT_SHA256
    ):
        raise FormalRunError(
            "protocol-v7 predecessor lacks the frozen protocol-v6 preflight"
        )
    predecessor_lineage = predecessor.get("lineage")
    previous = (
        predecessor_lineage.get("predecessor")
        if isinstance(predecessor_lineage, dict)
        else None
    )
    if (
        not isinstance(predecessor_lineage, dict)
        or predecessor_lineage.get("attempt_class")
        != "protocol_v6_recovery"
        or not isinstance(previous, dict)
        or previous.get("receipt_sha256") != _V5_RECOVERY_RECEIPT_SHA256
    ):
        raise FormalRunError(
            "protocol-v7 predecessor lacks exact protocol-v6 recovery lineage"
        )
    case = manifest.case(case_id)
    source = predecessor.get("source")
    expected_source = case["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in (
            "repository", "path", "symbol", "frontend", "file_sha256",
            "symbol_sha256",
        )
    ):
        raise FormalRunError("protocol-v7 predecessor source identity differs")
    if predecessor.get("provider") != manifest.data["provider"]:
        raise FormalRunError("protocol-v7 predecessor provider identity differs")
    return {
        "attempt_class": attempt_class,
        "predecessor": {
            "schema": predecessor["schema"],
            "batch_id": predecessor["batch_id"],
            "case_id": predecessor["case_id"],
            "status": predecessor["status"],
            "attempt_class": predecessor_lineage["attempt_class"],
            "manifest_digest": predecessor["manifest_digest"],
            "manifest_file_sha256": predecessor["manifest_file_sha256"],
            "receipt_sha256": _sha256(payload),
        },
    }


def _v2_lineage(
    manifest: BatchManifest,
    case_id: str,
    attempt_class: str | None,
    predecessor_receipt: str | os.PathLike[str] | None,
) -> Mapping[str, Any] | None:
    """Compatibility entry point for frozen v2 tests and callers."""

    return _lineage(manifest, case_id, attempt_class, predecessor_receipt)


def run_formal_gate1_case(
    manifest_path: str | os.PathLike[str],
    case_id: str,
    *,
    output_root: str | os.PathLike[str],
    preflight_path: str | os.PathLike[str],
    compiler_timeout: float = 300.0,
    attempt_class: str | None = None,
    predecessor_receipt: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Spend one Lift/Replay budget for one frozen case and run strict Gate1."""

    manifest_path = Path(manifest_path).resolve()
    manifest = load_batch_manifest(
        manifest_path, verify_sources=True, verify_repositories=True
    )
    if "lift_wire" in manifest.data:
        try:
            verify_lift_wire_lock(manifest.data["lift_wire"])
        except LiftWireError as exc:
            raise FormalRunError(str(exc)) from exc
    resolved_preflight = Path(preflight_path).resolve()
    preflight = _validate_preflight(manifest, resolved_preflight, manifest_path)
    case = manifest.case(case_id)
    case_dir = Path(output_root).resolve() / case_id
    receipt_path = case_dir / "receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") == "gate1_closed":
            return existing
        raise FormalRunError(
            "an incomplete formal ledger already exists; automatic retry is forbidden"
        )
    lineage = _lineage(
        manifest, case_id, attempt_class, predecessor_receipt
    )
    case_dir.mkdir(parents=True, exist_ok=False)
    source_path = resolve_source_path(manifest, case)
    source = source_path.read_text(encoding="utf-8")
    receipt = _initial_receipt(
        manifest,
        manifest_path,
        case,
        source_path,
        resolved_preflight,
        preflight,
        lineage,
    )
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
    if case["source"]["frontend"] == "triton":
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
    elif case["source"]["frontend"] == "cute":
        compiler = CuteCompiler(
            timeout=compiler_timeout,
            cache_dir=case_dir / "compiler_cache",
            arch="sm_120a",
        )
        runner = CuteGate1Runner(
            compiler=compiler,
            writer=writer,
            config=Gate1Config(observer_level=2, run_masked_j=True),
        )
        result = runner.run(
            artifact,
            source=CuteKernelSource.from_file(
                source_path,
                case["source"]["symbol"],
                cute_entry_symbol(source, case["source"]["symbol"]),
                trusted=True,
            ),
            source_case=cute_compile_case(case),
        )
    else:
        raise FormalRunError(
            f"no formal Gate1 runner for frontend {case['source']['frontend']!r}"
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
    parser.add_argument(
        "--attempt-class",
        choices=sorted(
            _V2_ATTEMPT_CLASSES | _V3_ATTEMPT_CLASSES | _V4_ATTEMPT_CLASSES
            | _V5_ATTEMPT_CLASSES
            | _V6_ATTEMPT_CLASSES
            | _V7_ATTEMPT_CLASSES
        ),
    )
    parser.add_argument("--predecessor-receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = run_formal_gate1_case(
        args.manifest,
        args.case_id,
        output_root=args.output_root,
        preflight_path=args.preflight,
        compiler_timeout=args.compiler_timeout,
        attempt_class=args.attempt_class,
        predecessor_receipt=args.predecessor_receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "gate1_closed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "FormalRunError",
    "LedgerReplayWriter",
    "RUN_SCHEMA",
    "RUN_SCHEMA_V2",
    "RUN_SCHEMA_V3",
    "RUN_SCHEMA_V4",
    "RUN_SCHEMA_V5",
    "RUN_SCHEMA_V6",
    "RUN_SCHEMA_V7",
    "run_formal_gate1_case",
]
