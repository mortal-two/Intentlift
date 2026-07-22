from __future__ import annotations

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.verify.gate1.model import (
    RealizationAuthorization,
    RefinementKind,
    SchemaError,
)
from intentir.verify.gate1.pi import RealizationWitness
from intentir.verify.gate1.prompt import build_normal_rewriter_view
from intentir.verify.typecheck import TypeCheckErrorCode, type_check


def _codes(artifact) -> set[TypeCheckErrorCode]:
    return {
        item.code
        for item in type_check(artifact.j, artifact.c, artifact.pi).diagnostics
    }


def test_valid_dependent_witness_passes_before_replay() -> None:
    artifact = make_reduce_artifact()
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed
    assert report.diagnostics == ()
    assert set(build_normal_rewriter_view(artifact)) == {"j", "c", "pi", "u_f"}


def test_dangling_anchor_fails_and_cannot_enter_replay() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][0]["semantic_ref"] = "n99"
    artifact.pi = RealizationWitness.from_json(payload)

    assert TypeCheckErrorCode.UNRESOLVED_ANCHOR in _codes(artifact)
    with pytest.raises(SchemaError, match="unresolved_anchor"):
        build_normal_rewriter_view(artifact)


def test_route_slot_and_carrier_type_are_checked_against_j() -> None:
    out_of_range = make_reduce_artifact()
    payload = out_of_range.pi.to_json()
    payload["routes"][0]["slots"] = [2]
    out_of_range.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_SLOT_OUT_OF_RANGE in _codes(out_of_range)

    wrong_type = make_reduce_artifact()
    payload = wrong_type.pi.to_json()
    payload["instances"][0]["semantic_ref"] = "p2"
    wrong_type.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_TYPE_MISMATCH in _codes(wrong_type)


def test_route_and_order_form_one_acyclic_dependency_graph() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["routes"].append({"source": "i3", "target": "i0", "slots": []})
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_CYCLE in _codes(artifact)


def test_c_value_type_is_not_an_executable_free_text_escape() -> None:
    artifact = make_reduce_artifact()
    artifact.c.values["c0"] = "code()"
    assert TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH in _codes(artifact)

    artifact.c.slots[0].value_type = "string"
    assert TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH in _codes(artifact)


def test_required_c_slot_must_have_the_actual_harness_valuation() -> None:
    artifact = make_reduce_artifact()
    artifact.c.values.clear()
    assert TypeCheckErrorCode.C_REQUIRED_VALUE_MISSING in _codes(artifact)


def test_observable_realization_cannot_be_owned_by_c() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][3]["semantic_ref"] = "c0"
    payload["instances"][3]["mechanism"] = "transfer_direct"
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.OBSERVABLE_REQUIRES_J_ANCHOR in _codes(artifact)


def test_sensitive_atomic_requires_exact_j_authorization() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][0]["mechanism"] = "update_atomic"
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_MISSING in _codes(artifact)

    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.NATIVE_ATOMIC, "a0")
    ]
    payload["refinements"] = [
        {"instance": "i0", "kind": "native_atomic", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert type_check(artifact.j, artifact.c, artifact.pi).passed


def test_sensitive_authorization_subject_must_match_refined_anchor() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.NATIVE_ATOMIC, "n0")
    ]
    payload = artifact.pi.to_json()
    payload["instances"][0]["mechanism"] = "update_atomic"
    payload["refinements"] = [
        {"instance": "i0", "kind": "native_atomic", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_SUBJECT_MISMATCH in _codes(artifact)


def test_typed_authorization_source_must_match_its_j_carrier() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization(
            "u0", RefinementKind.NUMERIC_CAST, "n0", "i32", "f32"
        )
    ]
    payload = artifact.pi.to_json()
    payload["refinements"] = [
        {"instance": "i1", "kind": "numeric_cast", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH in _codes(artifact)


def test_global_safe_refinement_is_closed_and_cannot_cite_candidate_auth() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.REASSOCIATION, "n0")
    ]
    payload = artifact.pi.to_json()
    payload["refinements"] = [
        {
            "instance": "i1",
            "kind": "identity_materialization",
            "auth_ref": "u0",
        }
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.GLOBAL_SAFE_AUTH_FORBIDDEN in _codes(artifact)
