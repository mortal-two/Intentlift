from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from intentir.evaluation.real_batch import (
    ManifestError,
    load_batch_manifest,
    probe_environment,
    scrc_compiler_view,
    triton_compile_case,
    validate_batch_manifest,
)
from intentir.evaluation.lift_wire import (
    LiftWireError,
    current_lift_wire_identity,
    verify_lift_wire_lock,
)
from intentir.evaluation.formal_preflight import (
    current_gate1_code_identity,
    verify_formal_preflight,
)


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "test" / "real_llm_batch_28.json"
_MANIFEST_V2 = _ROOT / "test" / "real_llm_batch_28_v2.json"
_MANIFEST_V3 = _ROOT / "test" / "real_llm_batch_28_v3.json"
_MANIFEST_V4 = _ROOT / "test" / "real_llm_batch_28_v4.json"
_MANIFEST_V5 = _ROOT / "test" / "real_llm_batch_28_v5.json"
_PREFLIGHT = _ROOT / "artifacts" / "real_llm_batch_28" / "preflight.json"
_PREFLIGHT_V2 = (
    _ROOT / "artifacts" / "real_llm_batch_28_v2" / "preflight.json"
)
_PREFLIGHT_V3 = (
    _ROOT / "artifacts" / "real_llm_batch_28_v3" / "preflight.json"
)
_PREFLIGHT_V4 = (
    _ROOT / "artifacts" / "real_llm_batch_28_v4" / "preflight.json"
)
_PREFLIGHT_V5 = (
    _ROOT / "artifacts" / "real_llm_batch_28_v5" / "preflight.json"
)


def _raw() -> dict:
    return json.loads(_MANIFEST.read_text())


def _raw_v2() -> dict:
    return json.loads(_MANIFEST_V2.read_text())


def _raw_v3() -> dict:
    return json.loads(_MANIFEST_V3.read_text())


def _raw_v4() -> dict:
    return json.loads(_MANIFEST_V4.read_text())


def _raw_v5() -> dict:
    return json.loads(_MANIFEST_V5.read_text())


def test_formal_manifest_has_28_verified_source_symbols() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    assert len(manifest.cases) == 28
    assert len(manifest.digest) == 64
    assert all(
        case["benchmark"]["threshold"]["maximum"] <= 1.10
        for case in manifest.cases
    )


def test_protocol_v2_is_self_contained_and_keeps_the_frozen_cases() -> None:
    v1 = load_batch_manifest(_MANIFEST)
    v2 = load_batch_manifest(_MANIFEST_V2)
    assert v1.digest == "12e2507433f23d6405cfa1dd0f1ff1169671a664822c07b0f5ecfbdc93570154"
    assert v2.digest == "486f18ce748f63afc4b7684fe2a7e5d1107da04c72a6282b7683851eb086f31e"
    assert v2.digest != v1.digest
    assert v2.data["batch_id"] != v1.data["batch_id"]
    assert v2.data["cases"] == v1.data["cases"]
    assert "base_manifest" not in v2.data
    assert len(v2.cases) == 28


def test_protocol_v2_lift_wire_lock_is_frozen_and_retired() -> None:
    manifest = load_batch_manifest(_MANIFEST_V2)
    receipt = json.loads(_PREFLIGHT_V2.read_text(encoding="utf-8"))
    assert receipt["lift_wire"] == manifest.data["lift_wire"]
    assert current_lift_wire_identity() != manifest.data["lift_wire"]
    with pytest.raises(LiftWireError, match="differs"):
        verify_lift_wire_lock(manifest.data["lift_wire"])


def test_protocol_v2_preflight_remains_complete_historical_evidence() -> None:
    manifest = load_batch_manifest(_MANIFEST_V2)
    receipt = json.loads(_PREFLIGHT_V2.read_text(encoding="utf-8"))
    assert receipt["manifest_digest"] == manifest.digest
    assert receipt["manifest_file_sha256"] == hashlib.sha256(
        _MANIFEST_V2.read_bytes()
    ).hexdigest()
    assert receipt["summary"] == {
        "blocking_capability": None,
        "cute_compiled": 2,
        "cute_total": 2,
        "overall_ready": True,
        "replay_arm_ready": 28,
        "source_observer_ready": 28,
        "source_pins_verified": 28,
        "symbol_pins_verified": 28,
        "triton_compiled": 26,
        "triton_total": 26,
    }
    # The v3 identity now also binds the strict parser and shared IntentOp
    # contract that the historical v2 preflight omitted.
    assert receipt["gate1_code"] != current_gate1_code_identity()
    cute_cases = [item for item in receipt["cases"] if item["frontend"] == "cute"]
    assert [item["async_copy_opcode"] for item in cute_cases] == [
        "cp.async.bulk.shared",
        "cp.async.bulk.tensor.1d.shared",
    ]


def test_current_gate1_identity_binds_parser_and_shared_contract() -> None:
    paths = {item["path"] for item in current_gate1_code_identity()}
    assert {
        "python/intentir/ir/intent_op_contract.py",
        "python/intentir/ir/model.py",
        "python/intentir/verify/gate1/model.py",
    } <= paths


def test_protocol_v2_preflight_cannot_be_reused_after_wire_epoch_drift() -> None:
    manifest = load_batch_manifest(_MANIFEST_V2)
    receipt = json.loads(_PREFLIGHT_V2.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Lift wire differs"):
        verify_formal_preflight(receipt, manifest, _MANIFEST_V2)


def test_protocol_v2_rejects_open_or_drifted_wire_metadata() -> None:
    raw = _raw_v2()
    raw["lift_wire"]["extra"] = "not-closed"
    with pytest.raises(ManifestError, match="unknown keys"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw_v2()
    raw["lift_wire"]["client_files"][0]["path"] = "../client.py"
    with pytest.raises(ManifestError, match="inside its repository root"):
        validate_batch_manifest(raw, verify_sources=False)

    expected = copy.deepcopy(_raw_v2()["lift_wire"])
    expected["prompt_sha256"] = "0" * 64
    with pytest.raises(LiftWireError, match="differs"):
        verify_lift_wire_lock(expected)


def test_protocol_v3_is_self_contained_retired_evidence() -> None:
    v2 = load_batch_manifest(_MANIFEST_V2)
    v3 = load_batch_manifest(_MANIFEST_V3)
    assert v3.digest == "5f0b1ee853e2821a299753165145023562df548810468c0110bdbc6895823222"
    assert v3.digest != v2.digest
    assert v3.data["cases"] == v2.data["cases"]
    assert "base_manifest" not in v3.data
    assert len(v3.cases) == 28
    assert v3.data["lift_wire"] != current_lift_wire_identity()
    with pytest.raises(LiftWireError, match="differs"):
        verify_lift_wire_lock(v3.data["lift_wire"])
    assert [item["path"] for item in v3.data["lift_wire"]["client_files"]][
        -1
    ] == "python/intentir/ir/intent_op_contract.py"


def test_protocol_v3_rejects_epoch_and_wire_tampering() -> None:
    raw = _raw_v3()
    raw["protocols"]["lift"] = "llm_direct_pi_source_aware_lift_v2"
    with pytest.raises(ManifestError, match="protocols"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw_v3()
    raw["lift_wire"]["version"] = "intentir.lift-wire.v2"
    with pytest.raises(ManifestError, match="version"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw_v3()
    raw["lift_wire"]["client_files"].pop()
    with pytest.raises(ManifestError, match="every request-path"):
        validate_batch_manifest(raw, verify_sources=False)


def test_protocol_v3_preflight_remains_complete_historical_evidence() -> None:
    manifest = load_batch_manifest(_MANIFEST_V3)
    receipt = json.loads(_PREFLIGHT_V3.read_text(encoding="utf-8"))
    assert receipt["manifest_digest"] == manifest.digest
    assert receipt["manifest_file_sha256"] == hashlib.sha256(
        _MANIFEST_V3.read_bytes()
    ).hexdigest()
    assert hashlib.sha256(_PREFLIGHT_V3.read_bytes()).hexdigest() == (
        "815d43b31635892e67aa3fddcd8f7843fffafa65d3064467b99c9afa5d3744f4"
    )
    assert receipt["schema"] == "intentir.real-llm-batch-preflight.v3"
    assert receipt["online_llm_calls"] == 0
    assert receipt["summary"] == {
        "blocking_capability": None,
        "cute_compiled": 2,
        "cute_total": 2,
        "overall_ready": True,
        "replay_arm_ready": 28,
        "source_observer_ready": 28,
        "source_pins_verified": 28,
        "symbol_pins_verified": 28,
        "triton_compiled": 26,
        "triton_total": 26,
    }


def test_protocol_v4_is_retired_ordered_and_non_c_safe() -> None:
    v3 = load_batch_manifest(_MANIFEST_V3)
    v4 = load_batch_manifest(_MANIFEST_V4)
    assert v4.digest == (
        "0da031b95049acf5884e1adb60126253ab5441d6f7de142a0373e2659ab98b18"
    )
    assert v4.digest != v3.digest
    assert len(v4.cases) == 28
    assert v4.data["lift_wire"] != current_lift_wire_identity()
    with pytest.raises(LiftWireError, match="differs"):
        verify_lift_wire_lock(v4.data["lift_wire"])
    case = v4.case("liger_swiglu_forward")
    parameters = case["specialization"]["parameters"]
    assert [item["name"] for item in parameters] == [
        "a_ptr", "b_ptr", "c_ptr", "stride", "n_cols", "BLOCK_SIZE"
    ]
    assert parameters[4] == {
        "name": "n_cols",
        "binding": "constexpr",
        "class": "binding",
        "value": 1024,
    }
    assert parameters[5] == {
        "name": "BLOCK_SIZE",
        "binding": "constexpr",
        "class": "schedule",
        "value": 1024,
    }
    assert [item["shape"] for item in parameters[:3]] == [
        [128, 1024], [128, 1024], [128, 1024]
    ]


def test_protocol_v4_rejects_open_or_reordered_parameter_classification() -> None:
    raw = _raw_v4()
    raw["cases"][0]["specialization"]["parameters"][4]["class"] = "semantic"
    with pytest.raises(ManifestError, match="binding or schedule"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw_v4()
    parameters = raw["cases"][0]["specialization"]["parameters"]
    parameters[4], parameters[5] = parameters[5], parameters[4]
    with pytest.raises(ManifestError, match="order differs"):
        validate_batch_manifest(raw, verify_sources=True)


def test_protocol_v4_preflight_remains_frozen_zero_llm_evidence() -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    receipt = json.loads(_PREFLIGHT_V4.read_text(encoding="utf-8"))
    assert hashlib.sha256(_PREFLIGHT_V4.read_bytes()).hexdigest() == (
        "74862a1402776d68988f3e704b2ceb22d61e89a5849e6a8c1da6c278e585fa30"
    )
    assert receipt["manifest_digest"] == manifest.digest
    assert receipt["lift_wire"] == manifest.data["lift_wire"]
    assert receipt["lift_wire"] != current_lift_wire_identity()
    assert receipt["gate1_code"] != current_gate1_code_identity()
    with pytest.raises(ValueError, match="Lift wire differs"):
        verify_formal_preflight(receipt, manifest, _MANIFEST_V4)
    assert receipt["schema"] == "intentir.real-llm-batch-preflight.v4"
    assert receipt["online_llm_calls"] == 0
    assert receipt["summary"] == {
        "blocking_capability": None,
        "cute_compiled": 2,
        "cute_total": 2,
        "overall_ready": True,
        "replay_arm_ready": 28,
        "source_observer_ready": 28,
        "source_pins_verified": 28,
        "symbol_pins_verified": 28,
        "triton_compiled": 26,
        "triton_total": 26,
    }
    swiglu = next(
        item for item in receipt["cases"]
        if item["case_id"] == "liger_swiglu_forward"
    )
    assert swiglu["ttir_sha256"] == (
        "92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb"
    )
    assert swiglu["ttgir_sha256"] == (
        "75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16"
    )


def test_protocol_v5_is_current_and_changes_only_protocol_identity() -> None:
    v4 = load_batch_manifest(_MANIFEST_V4)
    v5 = load_batch_manifest(_MANIFEST_V5)
    assert v5.digest == (
        "bf6716ba75c0087a506031101a5628f30c06c29d08dfdaf60ef5a0bfcea5ceeb"
    )
    assert hashlib.sha256(_MANIFEST_V5.read_bytes()).hexdigest() == (
        "98056f9035d839e5d51f8b209accf0806606b7fbab2bca283a302c50265716cf"
    )
    assert v5.data["cases"] == v4.data["cases"]
    assert v5.data["repositories"] == v4.data["repositories"]
    assert v5.data["call_budget"] == v4.data["call_budget"]
    assert {
        key: value for key, value in v5.data["protocols"].items() if key != "lift"
    } == {
        key: value for key, value in v4.data["protocols"].items() if key != "lift"
    }
    assert verify_lift_wire_lock(v5.data["lift_wire"]) == (
        current_lift_wire_identity()
    )
    assert v5.data["lift_wire"]["client_files"][-1]["path"] == (
        "python/intentir/verify/gate1/pi.py"
    )
    case = v5.case("liger_swiglu_forward")
    parameters = case["specialization"]["parameters"]
    assert parameters[4]["class"] == "binding"
    assert parameters[5]["class"] == "schedule"


def test_protocol_v5_rejects_v4_identity_or_open_parameter_class() -> None:
    raw = _raw_v5()
    raw["lift_wire"] = _raw_v4()["lift_wire"]
    with pytest.raises(ManifestError, match="version"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw_v5()
    raw["cases"][0]["specialization"]["parameters"][4]["class"] = "semantic"
    with pytest.raises(ManifestError, match="binding or schedule"):
        validate_batch_manifest(raw, verify_sources=False)


def test_protocol_v3_preflight_cannot_authorize_protocol_v4() -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    receipt = json.loads(_PREFLIGHT_V3.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="active epoch"):
        verify_formal_preflight(receipt, manifest, _MANIFEST_V4)


def test_protocol_v4_preflight_cannot_authorize_protocol_v5() -> None:
    manifest = load_batch_manifest(_MANIFEST_V5)
    receipt = json.loads(_PREFLIGHT_V4.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="active epoch"):
        verify_formal_preflight(receipt, manifest, _MANIFEST_V5)


def test_protocol_v2_preflight_cannot_authorize_protocol_v3() -> None:
    manifest = load_batch_manifest(_MANIFEST_V3)
    receipt = json.loads(_PREFLIGHT_V2.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="active epoch"):
        verify_formal_preflight(receipt, manifest, _MANIFEST_V3)


def test_preflight_receipt_is_bound_to_manifest_and_current_worker() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    receipt = json.loads(_PREFLIGHT.read_text())
    assert receipt["manifest_digest"] == manifest.digest
    assert receipt["manifest_file_sha256"] == hashlib.sha256(
        _MANIFEST.read_bytes()
    ).hexdigest()
    assert [item["case_id"] for item in receipt["cases"]] == [
        item["case_id"] for item in manifest.cases
    ]
    assert receipt["summary"]["triton_compiled"] == 26
    assert receipt["summary"]["cute_compiled"] == 2
    assert receipt["summary"]["overall_ready"] is True
    assert receipt["summary"]["blocking_capability"] is None
    worker = (
        _ROOT
        / "python"
        / "intentir"
        / "verify"
        / "gate1"
        / "_triton_worker.py"
    )
    assert receipt["compiler"]["gate1_worker_sha256"] == hashlib.sha256(
        worker.read_bytes()
    ).hexdigest()
    cute_worker = (
        _ROOT
        / "python"
        / "intentir"
        / "verify"
        / "gate1"
        / "_cute_worker.py"
    )
    assert receipt["compiler"]["cute_worker_sha256"] == hashlib.sha256(
        cute_worker.read_bytes()
    ).hexdigest()
    cute_cases = [item for item in receipt["cases"] if item["frontend"] == "cute"]
    assert [item["async_copy_opcode"] for item in cute_cases] == [
        "cp.async.bulk.shared",
        "cp.async.bulk.tensor.1d.shared",
    ]


def test_manifest_rejects_semantic_artifact_side_channel() -> None:
    raw = _raw()
    raw["cases"][0]["logical_contract"]["j"] = {"answer": "hidden"}
    with pytest.raises(ManifestError, match="banned semantic artifact"):
        validate_batch_manifest(raw, verify_sources=False)


def test_manifest_rejects_source_or_symbol_digest_drift() -> None:
    raw = _raw()
    raw["cases"][0]["source"]["symbol_sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="symbol digest changed"):
        validate_batch_manifest(raw)


def test_manifest_rejects_unbalanced_strata_and_call_budget() -> None:
    raw = _raw()
    raw["cases"][0]["stratum"] = raw["cases"][7]["stratum"]
    with pytest.raises(ManifestError, match="seven cases each"):
        validate_batch_manifest(raw, verify_sources=False)

    raw = _raw()
    raw["call_budget"]["recovery_per_case"] = 1
    with pytest.raises(ManifestError, match="call budget"):
        validate_batch_manifest(raw, verify_sources=False)


def test_manifest_rejects_specialization_abi_reordering() -> None:
    raw = _raw()
    arguments = raw["cases"][0]["specialization"]["arguments"]
    arguments[0], arguments[1] = arguments[1], arguments[0]
    with pytest.raises(ManifestError, match="specialization order"):
        validate_batch_manifest(raw)


def test_environment_probe_never_contains_provider_secret() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    rendered = json.dumps(probe_environment(manifest), sort_keys=True)
    config = _ROOT / "llm_providers.local.json"
    if config.exists():
        values = json.loads(config.read_text())
        for provider in values.values():
            secret = provider.get("api_key")
            if secret:
                assert secret not in rendered
    assert "api_key" not in rendered


def test_manifest_specialization_projects_to_gate1_compile_case() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    case = manifest.case("xformers_index_select_forward")
    projected = triton_compile_case(case)
    assert list(projected.arguments) == [
        "output_ptr",
        "source_ptr",
        "index_ptr",
        "num_indices",
        "num_cols",
        "stride0",
        "stride1",
    ]
    assert projected.grid == (128, 1)
    assert projected.schedule == {
        "BLOCK_SIZE_INDEX": 1,
        "BLOCK_SIZE_COL": 512,
        "num_warps": 4,
    }


def test_manifest_specialization_projects_to_source_aware_lift_view() -> None:
    manifest = load_batch_manifest(_MANIFEST)
    case = manifest.case("liger_swiglu_forward")
    view = scrc_compiler_view(case)
    assert view.parameter_order == (
        "a_ptr",
        "b_ptr",
        "c_ptr",
        "stride",
        "n_cols",
        "BLOCK_SIZE",
    )
    assert [item.name for item in view.arguments] == [
        "a_ptr",
        "b_ptr",
        "c_ptr",
        "stride",
    ]
    assert view.schedule_values == {
        "n_cols": 1024,
        "BLOCK_SIZE": 1024,
        "num_warps": 4,
    }
    assert view.compiler_slots == ("num_warps",)
    assert view.grid == (128,)
    assert view.grid_contract == ("128",)


def test_v4_specialization_separates_bound_constexpr_from_pure_c() -> None:
    manifest = load_batch_manifest(_MANIFEST_V4)
    case = manifest.case("liger_swiglu_forward")
    view = scrc_compiler_view(case)
    assert view.parameter_order == (
        "a_ptr", "b_ptr", "c_ptr", "stride", "n_cols", "BLOCK_SIZE"
    )
    assert view.bound_constexpr_values == {"n_cols": 1024}
    assert view.schedule_values == {"BLOCK_SIZE": 1024, "num_warps": 4}
    assert [item.shape for item in view.arguments[:3]] == [
        (128, 1024), (128, 1024), (128, 1024)
    ]
    projected = triton_compile_case(case)
    assert projected.schedule == {
        "n_cols": 1024,
        "BLOCK_SIZE": 1024,
        "num_warps": 4,
    }


def test_required_anchor_must_resolve_to_case() -> None:
    raw = copy.deepcopy(_raw())
    raw["required_anchors"].append("not_a_case")
    with pytest.raises(ManifestError, match="required anchors"):
        validate_batch_manifest(raw, verify_sources=False)
