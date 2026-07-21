from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import intentir.evaluation.formal_runner as formal_runner
from intentir.evaluation.formal_runner import (
    FormalRunError,
    LedgerReplayWriter,
)
from intentir.evaluation.real_batch import (
    cute_entry_symbol,
    load_batch_manifest,
    resolve_source_path,
    scrc_compiler_view,
)
from intentir.extraction.lifters.scrc import prepare_scrc_lift
from intentir.extraction.providers import client
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.prompt import (
    build_rewriter_prompt,
    replay_packet_digest,
)
from intentir.verify.gate1.replay import ReplayGeneration

from direct_fixtures import make_copy_artifact


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "test" / "real_llm_batch_28.json"
_MANIFEST_V2 = _ROOT / "test" / "real_llm_batch_28_v2.json"
_PREFLIGHT = _ROOT / "artifacts" / "real_llm_batch_28" / "preflight.json"
_PRIMARY = _ROOT / "artifacts" / "real_llm_batch_28" / "primary"


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload


def _provider() -> ProviderConfig:
    return ProviderConfig(
        "formal-test",
        "https://provider.invalid/v1",
        "SECRET_MUST_NOT_ENTER_EVIDENCE",
        "test-model",
        reasoning_effort="xhigh",
    )


def test_provider_reply_binds_exact_secret_free_wire_bytes(monkeypatch) -> None:
    raw = json.dumps(
        {
            "id": "resp-test",
            "model": "returned-model",
            "output_text": "answer",
            "usage": {"total_tokens": 3},
        }
    ).encode()
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return _Response(raw)

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    reply = client.call_responses(
        "instructions",
        "input",
        provider=_provider(),
        effort="xhigh",
    )
    assert reply.request_sha256 == hashlib.sha256(captured["body"]).hexdigest()
    assert reply.response_sha256 == hashlib.sha256(b"answer").hexdigest()
    assert reply.raw_response_sha256 == hashlib.sha256(raw).hexdigest()
    assert reply.raw_response == raw
    assert reply.attempts == 1
    assert reply.elapsed_ms >= 0
    assert b"SECRET_MUST_NOT_ENTER_EVIDENCE" not in captured["body"]
    assert captured["authorization"].endswith("SECRET_MUST_NOT_ENTER_EVIDENCE")


def test_replay_ledger_writes_before_call_and_never_contains_source(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = make_copy_artifact()
    receipt = {
        "call_budget": {"normal_replay": 1, "masked_j": 1},
        "calls": [],
        "history": [],
        "status": "lift_closed",
    }
    writer = LedgerReplayWriter(
        provider=_provider(),
        effort="xhigh",
        case_dir=tmp_path,
        receipt=receipt,
    )

    def fake_generate(value, *, masked=False):
        instructions, user = build_rewriter_prompt(value, masked=masked)
        body = client.responses_request_body(
            instructions,
            user,
            provider=_provider(),
            effort="xhigh",
        )
        packet = json.loads(user)
        return ReplayGeneration(
            code="import triton\nimport triton.language as tl\n\n@triton.jit\ndef scrc_k0(p0, p1, p2, c0: tl.constexpr):\n    pass\n",
            function_name="scrc_k0",
            model="test-model",
            response_id="resp0",
            packet_digest=replay_packet_digest(packet),
            prompt_digest=hashlib.sha256((instructions + "\0" + user).encode()).hexdigest(),
            request_sha256=hashlib.sha256(body).hexdigest(),
            response_sha256=hashlib.sha256(b"raw replay").hexdigest(),
            raw_response_sha256=hashlib.sha256(b"{}").hexdigest(),
            response_text="raw replay",
            raw_response=b"{}",
            attempts=1,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(writer.inner, "generate", fake_generate)
    generation = writer.generate(artifact)
    assert generation.response_id == "resp0"
    assert receipt["calls"][0]["status"] == "completed"
    request = (tmp_path / receipt["calls"][0]["request_path"]).read_text()
    assert "PRIVATE_SOURCE_CANARY" not in request
    assert "source_code" not in request
    assert (tmp_path / receipt["calls"][0]["replay_code_path"]).is_file()
    with pytest.raises(FormalRunError, match="already present"):
        writer.generate(artifact)


def test_formal_cute_case_has_generic_lift_and_compile_bindings() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    case = manifest.case("cute_async_tma_copy")
    view = scrc_compiler_view(case)
    assert [item.kind for item in view.arguments] == [
        "tensor",
        "opaque",
        "tensor",
    ]
    assert view.arguments[1].dtype == "cute.CopyAtom"
    assert view.arguments[1].value is None
    source = resolve_source_path(manifest, case).read_text(encoding="utf-8")
    lift_input = json.loads(
        prepare_scrc_lift(
            source,
            compiler_view=view,
            source_symbol=case["source"]["symbol"],
        ).user_input
    )
    opaque_argument = lift_input["harness_compiler_view"]["arguments"][1]
    assert opaque_argument == {
        "name": "p1",
        "kind": "opaque",
        "dtype": "cute.CopyAtom",
    }
    assert "f32_tma_tile_128" not in json.dumps(lift_input)
    assert cute_entry_symbol(source, case["source"]["symbol"]) == "go"


def test_formal_v2_rejects_wire_drift_before_provider_or_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V2)
    data = json.loads(json.dumps(manifest.data))
    data["lift_wire"]["schema_sha256"] = "0" * 64
    drifted = type(manifest)(data, manifest.source_root, manifest.digest)
    monkeypatch.setattr(formal_runner, "load_batch_manifest", lambda *args, **kwargs: drifted)

    def provider_must_not_run(*args, **kwargs):
        raise AssertionError("provider resolution happened before wire verification")

    monkeypatch.setattr(formal_runner, "get_provider", provider_must_not_run)
    output = tmp_path / "formal"
    with pytest.raises(FormalRunError, match="Lift wire differs"):
        formal_runner.run_formal_gate1_case(
            _MANIFEST_V2,
            "liger_swiglu_forward",
            output_root=output,
            preflight_path=tmp_path / "missing-preflight.json",
        )
    assert not output.exists()


def test_protocol_v2_recovery_lineage_binds_the_failed_v1_receipt() -> None:
    manifest = load_batch_manifest(_MANIFEST_V2)
    predecessor = _PRIMARY / "liger_swiglu_forward" / "receipt.json"
    lineage = formal_runner._v2_lineage(
        manifest,
        "liger_swiglu_forward",
        "protocol_v2_recovery",
        predecessor,
    )
    assert lineage is not None
    assert lineage["attempt_class"] == "protocol_v2_recovery"
    assert lineage["predecessor"] == {
        "batch_id": "tianchenir-formal-real-kernels-2026-07-22",
        "case_id": "liger_swiglu_forward",
        "status": "lift_failed",
        "receipt_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
    }
    with pytest.raises(FormalRunError, match="requires its predecessor"):
        formal_runner._v2_lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v2_recovery",
            None,
        )


def test_committed_first_attempt_is_complete_and_not_rewritten_as_success() -> None:
    case_dir = _PRIMARY / "liger_swiglu_forward"
    receipt = json.loads((case_dir / "receipt.json").read_text())
    assert receipt["status"] == "lift_failed"
    assert len(receipt["calls"]) == 1
    call = receipt["calls"][0]
    assert call["kind"] == "lift"
    assert call["status"] == "failed"
    assert call["attempts"] == 1
    assert call["usage"] == {
        "input_tokens": 8434,
        "output_tokens": 14235,
        "total_tokens": 22669,
    }
    for path_key, digest_key in (
        ("request_path", "request_sha256"),
        ("response_path", "response_sha256"),
        ("raw_response_path", "raw_response_sha256"),
    ):
        payload = (case_dir / call[path_key]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == call[digest_key]
    request = (case_dir / call["request_path"]).read_text()
    assert "_swiglu_forward_kernel" not in request
    assert '"kind":"normal_replay"' not in request
