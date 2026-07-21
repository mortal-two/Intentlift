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
from intentir.extraction.lifters.scrc import SCRCLiftAttempt, SCRCLiftError
from intentir.extraction.providers import client
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.prompt import (
    build_rewriter_prompt,
    replay_packet_digest,
)
from intentir.verify.gate1.model import Gate1Artifact, SchemaError
from intentir.verify.gate1.replay import ReplayGeneration

from direct_fixtures import make_copy_artifact


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "test" / "real_llm_batch_28.json"
_MANIFEST_V2 = _ROOT / "test" / "real_llm_batch_28_v2.json"
_MANIFEST_V3 = _ROOT / "test" / "real_llm_batch_28_v3.json"
_MANIFEST_V4 = _ROOT / "test" / "real_llm_batch_28_v4.json"
_MANIFEST_V5 = _ROOT / "test" / "real_llm_batch_28_v5.json"
_PREFLIGHT = _ROOT / "artifacts" / "real_llm_batch_28" / "preflight.json"
_PREFLIGHT_V3 = _ROOT / "artifacts" / "real_llm_batch_28_v3" / "preflight.json"
_PREFLIGHT_V4 = _ROOT / "artifacts" / "real_llm_batch_28_v4" / "preflight.json"
_PREFLIGHT_V5 = _ROOT / "artifacts" / "real_llm_batch_28_v5" / "preflight.json"
_PRIMARY = _ROOT / "artifacts" / "real_llm_batch_28" / "primary"
_RECOVERY_V2 = _ROOT / "artifacts" / "real_llm_batch_28_v2" / "recovery"
_PROGRESS_V2 = _ROOT / "artifacts" / "real_llm_batch_28_v2" / "progress.json"
_RECOVERY_V3 = _ROOT / "artifacts" / "real_llm_batch_28_v3" / "recovery"
_PROGRESS_V3 = _ROOT / "artifacts" / "real_llm_batch_28_v3" / "progress.json"
_RECOVERY_V4 = _ROOT / "artifacts" / "real_llm_batch_28_v4" / "recovery"


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


def test_v4_swiglu_lift_view_preserves_non_c_and_row_topology_inputs() -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    case = manifest.case("liger_swiglu_forward")
    source = resolve_source_path(manifest, case).read_text(encoding="utf-8")
    request = prepare_scrc_lift(
        source,
        compiler_view=scrc_compiler_view(case),
        source_symbol=case["source"]["symbol"],
    )
    lift_input = json.loads(request.user_input)
    view = lift_input["harness_compiler_view"]
    assert "def scrc_k0(p0, p1, p2, p3, p4: tl.constexpr, c0: tl.constexpr)" in (
        lift_input["anonymous_source"]
    )
    assert view["bound_constexpr_values"] == {"p4": 1024}
    assert view["schedule_values"] == {"c0": 1024, "c1": 4}
    assert [item.get("shape") for item in view["arguments"][:3]] == [
        [128, 1024], [128, 1024], [128, 1024]
    ]
    assert view["parameter_order"] == [
        "p0", "p1", "p2", "p3", "p4", "c0"
    ]


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
        "schema": "intentir.formal-real-llm-case.v1",
        "batch_id": "tianchenir-formal-real-kernels-2026-07-22",
        "case_id": "liger_swiglu_forward",
        "status": "lift_failed",
        "attempt_class": None,
        "manifest_digest": "12e2507433f23d6405cfa1dd0f1ff1169671a664822c07b0f5ecfbdc93570154",
        "manifest_file_sha256": "a84ea446df84f393bab0aa25eebfd183bbf4b7d9144d0bc8c4c86eeaa7fe09f9",
        "receipt_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
    }
    with pytest.raises(FormalRunError, match="requires its predecessor"):
        formal_runner._v2_lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v2_recovery",
            None,
        )


def test_protocol_v3_recovery_binds_exact_frozen_v2_failure() -> None:
    manifest = load_batch_manifest(_MANIFEST_V3)
    predecessor = _RECOVERY_V2 / "liger_swiglu_forward" / "receipt.json"
    lineage = formal_runner._lineage(
        manifest,
        "liger_swiglu_forward",
        "protocol_v3_recovery",
        predecessor,
    )
    assert lineage is not None
    assert lineage["attempt_class"] == "protocol_v3_recovery"
    assert lineage["predecessor"] == {
        "schema": "intentir.formal-real-llm-case.v2",
        "batch_id": "tianchenir-formal-real-kernels-protocol-v2-2026-07-22",
        "case_id": "liger_swiglu_forward",
        "status": "lift_failed",
        "attempt_class": "protocol_v2_recovery",
        "manifest_digest": "486f18ce748f63afc4b7684fe2a7e5d1107da04c72a6282b7683851eb086f31e",
        "manifest_file_sha256": "27b9a81e8230ce75105848625b85f44e9002264c91d166e57983c617e09e83b3",
        "receipt_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
    }
    with pytest.raises(FormalRunError, match="requires its predecessor"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v3_recovery",
            None,
        )
    with pytest.raises(FormalRunError, match="explicit attempt classification"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v2_recovery",
            predecessor,
        )


def test_protocol_v4_recovery_binds_exact_v3_gate1_failure() -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    predecessor = _RECOVERY_V3 / "liger_swiglu_forward" / "receipt.json"
    lineage = formal_runner._lineage(
        manifest,
        "liger_swiglu_forward",
        "protocol_v4_recovery",
        predecessor,
    )
    assert lineage == {
        "attempt_class": "protocol_v4_recovery",
        "predecessor": {
            "schema": "intentir.formal-real-llm-case.v3",
            "batch_id": "tianchenir-formal-real-kernels-protocol-v3-2026-07-22",
            "case_id": "liger_swiglu_forward",
            "status": "gate1_failed",
            "attempt_class": "protocol_v3_recovery",
            "manifest_digest": "5f0b1ee853e2821a299753165145023562df548810468c0110bdbc6895823222",
            "manifest_file_sha256": "2528ae016445afd63f9c0083213ab2594b97756bcd7b1950edb28e1273212205",
            "receipt_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
        },
    }
    with pytest.raises(FormalRunError, match="requires its predecessor"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v4_recovery",
            None,
        )


def test_protocol_v5_recovery_binds_exact_v4_pi_endpoint_rejection() -> None:
    manifest = load_batch_manifest(_MANIFEST_V5)
    predecessor = _RECOVERY_V4 / "liger_swiglu_forward" / "receipt.json"
    lineage = formal_runner._lineage(
        manifest,
        "liger_swiglu_forward",
        "protocol_v5_recovery",
        predecessor,
    )
    assert lineage == {
        "attempt_class": "protocol_v5_recovery",
        "predecessor": {
            "schema": "intentir.formal-real-llm-case.v4",
            "batch_id": "tianchenir-formal-real-kernels-protocol-v4-2026-07-22",
            "case_id": "liger_swiglu_forward",
            "status": "lift_failed",
            "attempt_class": "protocol_v4_recovery",
            "manifest_digest": "0da031b95049acf5884e1adb60126253ab5441d6f7de142a0373e2659ab98b18",
            "manifest_file_sha256": "3d8be701069c404cce043779ca953113ec3c8cf2655dd1df6140430cebe800a9",
            "receipt_sha256": "98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2",
        },
    }
    with pytest.raises(FormalRunError, match="requires its predecessor"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v5_recovery",
            None,
        )


def test_protocol_v5_recovery_rejects_nonexact_v4_receipt(
    tmp_path: Path,
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V5)
    original = _RECOVERY_V4 / "liger_swiglu_forward" / "receipt.json"
    receipt = json.loads(original.read_text(encoding="utf-8"))
    receipt["history"].append({"status": "tampered", "at": "never"})
    predecessor = tmp_path / "receipt.json"
    predecessor.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FormalRunError, match="exact frozen protocol-v4 receipt"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v5_recovery",
            predecessor,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update({"status": "gate1_failed"}), "Lift failure"),
        (
            lambda receipt: receipt["calls"].append(
                {"kind": "normal_replay", "status": "failed", "attempts": 1}
            ),
            "one-call pi endpoint rejection",
        ),
        (
            lambda receipt: receipt["preflight"].update({"sha256": "0" * 64}),
            "frozen protocol-v4 preflight",
        ),
        (
            lambda receipt: receipt["lineage"].update(
                {"attempt_class": "protocol_v4_first_attempt"}
            ),
            "exact protocol-v4 recovery lineage",
        ),
        (
            lambda receipt: receipt["source"].update({"symbol_sha256": "0" * 64}),
            "source identity differs",
        ),
        (
            lambda receipt: receipt["provider"].update({"model": "other"}),
            "provider identity differs",
        ),
    ],
)
def test_protocol_v5_structurally_rejects_tampered_v4_predecessor(
    tmp_path: Path,
    monkeypatch,
    mutation,
    message: str,
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V5)
    original = _RECOVERY_V4 / "liger_swiglu_forward" / "receipt.json"
    receipt = json.loads(original.read_text(encoding="utf-8"))
    mutation(receipt)
    payload = json.dumps(receipt, sort_keys=True).encode()
    predecessor = tmp_path / "receipt.json"
    predecessor.write_bytes(payload)
    monkeypatch.setattr(
        formal_runner,
        "_V4_RECOVERY_RECEIPT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(FormalRunError, match=message):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v5_recovery",
            predecessor,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update({"status": "lift_failed"}), "frozen protocol-v3"),
        (lambda receipt: receipt["calls"].pop(), "one Lift and one normal Replay"),
        (lambda receipt: receipt["gate1"].update({"masked_j_calls": 1}), "strict Gate1"),
        (lambda receipt: receipt.update({"manifest_digest": "0" * 64}), "frozen protocol-v3"),
    ],
)
def test_protocol_v4_recovery_rejects_tampered_v3_predecessor(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    original = _RECOVERY_V3 / "liger_swiglu_forward" / "receipt.json"
    receipt = json.loads(original.read_text(encoding="utf-8"))
    mutation(receipt)
    predecessor = tmp_path / "receipt.json"
    predecessor.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FormalRunError, match=message):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v4_recovery",
            predecessor,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update({"status": "gate1_failed"}), "prior failed"),
        (
            lambda receipt: receipt["calls"].append(
                {"kind": "normal_replay", "status": "failed", "attempts": 1}
            ),
            "one auditable failed call",
        ),
        (
            lambda receipt: receipt.update({"manifest_digest": "0" * 64}),
            "frozen protocol-v2 recovery",
        ),
        (
            lambda receipt: receipt["source"].update({"symbol_sha256": "0" * 64}),
            "source identity differs",
        ),
    ],
)
def test_protocol_v3_recovery_rejects_tampered_predecessor(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V3)
    original = _RECOVERY_V2 / "liger_swiglu_forward" / "receipt.json"
    receipt = json.loads(original.read_text(encoding="utf-8"))
    mutation(receipt)
    predecessor = tmp_path / "receipt.json"
    predecessor.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FormalRunError, match=message):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v3_recovery",
            predecessor,
        )


def test_retired_protocol_v3_still_rejects_v2_attempt_class() -> None:
    manifest = load_batch_manifest(_MANIFEST_V3)
    with pytest.raises(FormalRunError, match="explicit attempt classification"):
        formal_runner._lineage(
            manifest,
            "liger_swiglu_forward",
            "protocol_v2_recovery",
            _RECOVERY_V2 / "liger_swiglu_forward" / "receipt.json",
        )


def test_protocol_v4_rejects_wrong_attempt_before_provider_or_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    def provider_must_not_run(*args, **kwargs):
        raise AssertionError("provider resolution happened before lineage validation")

    monkeypatch.setattr(formal_runner, "get_provider", provider_must_not_run)
    monkeypatch.setattr(formal_runner, "verify_lift_wire_lock", lambda _lock: None)
    monkeypatch.setattr(
        formal_runner,
        "_validate_preflight",
        lambda *_args, **_kwargs: {"schema": "offline-test"},
    )
    output = tmp_path / "formal"
    with pytest.raises(FormalRunError, match="explicit attempt classification"):
        formal_runner.run_formal_gate1_case(
            _MANIFEST_V4,
            "liger_swiglu_forward",
            output_root=output,
            preflight_path=_PREFLIGHT_V4,
            attempt_class="protocol_v3_recovery",
            predecessor_receipt=(
                _RECOVERY_V3 / "liger_swiglu_forward" / "receipt.json"
            ),
        )
    assert not output.exists()


def test_protocol_v5_rejects_wrong_attempt_before_provider_or_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    def provider_must_not_run(*args, **kwargs):
        raise AssertionError("provider resolution happened before lineage validation")

    monkeypatch.setattr(formal_runner, "get_provider", provider_must_not_run)
    output = tmp_path / "formal"
    with pytest.raises(FormalRunError, match="explicit attempt classification"):
        formal_runner.run_formal_gate1_case(
            _MANIFEST_V5,
            "liger_swiglu_forward",
            output_root=output,
            preflight_path=_PREFLIGHT_V5,
            attempt_class="protocol_v4_recovery",
            predecessor_receipt=(
                _RECOVERY_V4 / "liger_swiglu_forward" / "receipt.json"
            ),
        )
    assert not output.exists()


def test_invalid_lift_writes_one_failed_call_and_runs_no_downstream_arm(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    provider_spec = manifest.data["provider"]
    provider = ProviderConfig(
        provider_spec["name"],
        "https://provider.invalid/v1",
        "SECRET",
        provider_spec["model"],
        reasoning_effort=provider_spec["reasoning_effort"],
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    invalid_response = (
        _RECOVERY_V4
        / "liger_swiglu_forward"
        / "calls"
        / "01_lift"
        / "response.txt"
    ).read_text(encoding="utf-8")

    monkeypatch.setattr(formal_runner, "verify_lift_wire_lock", lambda _lock: None)
    monkeypatch.setattr(
        formal_runner,
        "_validate_preflight",
        lambda *_args, **_kwargs: {"schema": "offline-test"},
    )
    monkeypatch.setattr(formal_runner, "get_provider", lambda _name: provider)

    def reject_lift(*_args, **_kwargs):
        raise SCRCLiftError(
            SCRCLiftAttempt(
                model=provider.model,
                diagnostic=(
                    "SchemaError: pi.routes[0].source must reference a declared "
                    "pi instance id i*; got 'c1'"
                ),
                response_text=invalid_response,
                response_id="resp-offline-invalid",
                response_model=provider.model,
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                response_sha256=hashlib.sha256(
                    invalid_response.encode()
                ).hexdigest(),
                raw_response=b"{}",
                raw_response_sha256=hashlib.sha256(b"{}").hexdigest(),
                attempts=1,
                elapsed_ms=1.0,
            )
        )

    monkeypatch.setattr(formal_runner, "extract_scrc_artifact", reject_lift)
    output = tmp_path / "formal"
    with pytest.raises(SCRCLiftError, match=r"pi\.routes\[0\]\.source"):
        formal_runner.run_formal_gate1_case(
            _MANIFEST_V4,
            "liger_swiglu_forward",
            output_root=output,
            preflight_path=preflight,
            attempt_class="protocol_v4_recovery",
            predecessor_receipt=(
                _RECOVERY_V3 / "liger_swiglu_forward" / "receipt.json"
            ),
        )

    case_dir = output / "liger_swiglu_forward"
    receipt = json.loads((case_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "lift_failed"
    assert [
        (item["kind"], item["status"], item["attempts"])
        for item in receipt["calls"]
    ] == [("lift", "failed", 1)]
    assert receipt["artifacts"] == {}
    assert "gate1" not in receipt
    assert not (case_dir / "calls" / "02_normal_replay").exists()
    assert not (case_dir / "calls" / "02_masked_j").exists()


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


def test_committed_protocol_v2_recovery_failure_is_complete_and_stopped() -> None:
    case_dir = _RECOVERY_V2 / "liger_swiglu_forward"
    receipt_bytes = (case_dir / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert hashlib.sha256(receipt_bytes).hexdigest() == (
        "82029abd631873847de48babdb1c9ebfad3b3006585fd95d8053f19d4d6e0901"
    )
    assert receipt["schema"] == "intentir.formal-real-llm-case.v2"
    assert receipt["status"] == "lift_failed"
    assert receipt["lineage"]["attempt_class"] == "protocol_v2_recovery"
    assert len(receipt["calls"]) == 1
    call = receipt["calls"][0]
    assert call["kind"] == "lift"
    assert call["status"] == "failed"
    assert call["attempts"] == 1
    assert call["usage"] == {
        "input_tokens": 8713,
        "output_tokens": 10437,
        "total_tokens": 19150,
    }
    assert "non-loop operation 'body[0]'" in call["diagnostic"]
    for path_key, digest_key in (
        ("request_path", "request_sha256"),
        ("response_path", "response_sha256"),
        ("raw_response_path", "raw_response_sha256"),
    ):
        payload = (case_dir / call[path_key]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == call[digest_key]
    raw = json.loads((case_dir / call["raw_response_path"]).read_text())
    assert raw["status"] == "completed"
    assert raw["usage"]["output_tokens_details"]["reasoning_tokens"] == 8286
    request = (case_dir / call["request_path"]).read_text()
    assert "_swiglu_forward_kernel" not in request
    assert not (case_dir / "calls" / "02_normal_replay").exists()
    assert not (case_dir / "calls" / "02_masked_j").exists()

    progress = json.loads(_PROGRESS_V2.read_text())
    assert progress["online_http_attempts"] == 1
    assert progress["calls"] == {
        "lift": 1,
        "normal_replay": 0,
        "masked_j": 0,
    }
    assert progress["batch_stopped"] is True
    assert progress["unattempted_cases"] == 27


def test_committed_protocol_v3_canary_closes_lift_then_fails_strict_gate1() -> None:
    case_dir = _RECOVERY_V3 / "liger_swiglu_forward"
    receipt_bytes = (case_dir / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert hashlib.sha256(receipt_bytes).hexdigest() == (
        "ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1"
    )
    assert receipt["schema"] == "intentir.formal-real-llm-case.v3"
    assert receipt["status"] == "gate1_failed"
    assert receipt["lineage"]["attempt_class"] == "protocol_v3_recovery"
    assert receipt["lineage"]["predecessor"]["receipt_sha256"] == (
        "82029abd631873847de48babdb1c9ebfad3b3006585fd95d8053f19d4d6e0901"
    )
    assert [(call["kind"], call["status"], call["attempts"]) for call in receipt["calls"]] == [
        ("lift", "completed", 1),
        ("normal_replay", "completed", 1),
    ]
    assert [call["usage"] for call in receipt["calls"]] == [
        {"input_tokens": 9250, "output_tokens": 9793, "total_tokens": 19043},
        {"input_tokens": 2331, "output_tokens": 1635, "total_tokens": 3966},
    ]
    for call in receipt["calls"]:
        request = (case_dir / call["request_path"]).read_text()
        assert "_swiglu_forward_kernel" not in request
        assert "src/liger_kernel/ops/swiglu.py" not in request
    for call in receipt["calls"]:
        for path_key, digest_key in (
            ("request_path", "request_sha256"),
            ("response_path", "response_sha256"),
            ("raw_response_path", "raw_response_sha256"),
        ):
            payload = (case_dir / call[path_key]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == call[digest_key]
    lift_raw = json.loads(
        (case_dir / receipt["calls"][0]["raw_response_path"]).read_text()
    )
    replay_raw = json.loads(
        (case_dir / receipt["calls"][1]["raw_response_path"]).read_text()
    )
    assert lift_raw["usage"]["output_tokens_details"]["reasoning_tokens"] == 8061
    assert replay_raw["usage"]["output_tokens_details"]["reasoning_tokens"] == 1447

    artifact = Gate1Artifact.from_json(
        json.loads((case_dir / "lift" / "artifact.json").read_text())
    )
    assert artifact.j.intent.name == "scrc_j0"
    assert artifact.pi.instances
    assert receipt["gate1"] == {
        "artifact_sha256": "d7d4de8720edcfe6d8c75475c8bcccb55134c5309af8a65520b29216524188f6",
        "masked_j_calls": 0,
        "normal_calls": 1,
        "passed": False,
        "protocol": "llm_direct_pi_gate1_level2_v1",
        "receipt_path": "gate1/receipt.json",
        "receipt_sha256": "4f48345f32e2da27b5cd806a7d25b8d8344bc7005f1df9b64daa487c393bc135",
        "status": "no_closure",
    }
    assert not (case_dir / "calls" / "03_masked_j").exists()

    gate1 = json.loads((case_dir / "gate1" / "receipt.json").read_text())
    components = {
        item["category"]: item for item in gate1["normal"]["comparison"]["components"]
    }
    assert gate1["normal"]["comparison"]["score"] == pytest.approx(
        0.42689393939393944
    )
    assert components["interface"]["passed"] is True
    assert components["schedule-only"]["passed"] is True
    assert components["dataflow"]["passed"] is False
    assert len(components["dataflow"]["differences"]) == 81
    assert len(components["predicate_effect"]["differences"]) == 6
    assert {
        (item["item"], item["expected_count"], item["actual_count"])
        for item in components["op_multiset"]["differences"]
    } == {
        ("arith.addi", 0, 1),
        ("arith.constant", 4, 5),
        ("arith.extsi", 2, 0),
        ("tt.addptr", 6, 3),
        ("tt.splat", 3, 4),
    }

    progress = json.loads(_PROGRESS_V3.read_text())
    assert progress["online_http_attempts"] == 2
    assert progress["calls"] == {
        "lift": 1,
        "normal_replay": 1,
        "masked_j": 0,
    }
    assert progress["usage"]["reasoning_tokens"] == 9508
    assert progress["cases"][0]["failure_class"] == (
        "pi_realization_topology_incomplete"
    )
    assert progress["batch_stopped"] is True
    assert progress["unattempted_cases"] == 27


def test_committed_protocol_v4_canary_rejects_pi_endpoint_and_stops() -> None:
    case_dir = _RECOVERY_V4 / "liger_swiglu_forward"
    receipt_bytes = (case_dir / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert hashlib.sha256(receipt_bytes).hexdigest() == (
        "98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2"
    )
    assert receipt["schema"] == "intentir.formal-real-llm-case.v4"
    assert receipt["status"] == "lift_failed"
    assert receipt["lineage"]["attempt_class"] == "protocol_v4_recovery"
    assert receipt["lineage"]["predecessor"]["receipt_sha256"] == (
        "ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1"
    )
    assert [
        (item["kind"], item["status"], item["attempts"])
        for item in receipt["calls"]
    ] == [("lift", "failed", 1)]
    call = receipt["calls"][0]
    assert call["usage"] == {
        "input_tokens": 9555,
        "output_tokens": 14152,
        "total_tokens": 23707,
    }
    assert call["response_sha256"] == (
        "289aa3b37fb04c198941f31f76e31a4d05b118049dcf149e4593e8bde8e7db0a"
    )
    response = json.loads((case_dir / call["response_path"]).read_text())
    instance_ids = {item["id"] for item in response["pi"]["instances"]}
    invalid_route_sources = [
        item["source"]
        for item in response["pi"]["routes"]
        if item["source"] not in instance_ids
    ]
    invalid_sharing_sources = [
        item["source"]
        for item in response["pi"]["sharing"]
        if item["source"] not in instance_ids
    ]
    assert invalid_route_sources == [
        "c1", "p0", "p3", "p1", "p3", "p2", "p3", "c0", "p4"
    ]
    assert invalid_sharing_sources == ["p3"]
    with pytest.raises(
        SchemaError,
        match=(
            r"pi\.routes\[0\]\.source must reference a declared pi instance "
            r"id i\*; got 'c1'"
        ),
    ):
        Gate1Artifact.from_json(response)
    assert receipt["artifacts"] == {}
    assert "gate1" not in receipt
    assert not (case_dir / "calls" / "02_normal_replay").exists()
    assert not (case_dir / "calls" / "02_masked_j").exists()
