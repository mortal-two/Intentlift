from __future__ import annotations

import copy

import pytest

from intentir.backend.cuda.runtime_contract import RuntimeContract, RuntimePredicate
from intentir.backend.cuda.variant_manifest import (
    ArtifactProvenance,
    DeviceLaunchStep,
    ExecutableLaunchPlan,
    InvocationSlot,
    VariantManifest,
    VariantManifestError,
    VariantRecord,
)


def _artifact(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        intent_mlir_paths=(f"variants/{name}/intent.mlir",),
        binary_paths=(f"variants/{name}/kernel.cubin",),
        binary_modules=(name,),
        entry_points=(f"{name}_kernel",),
        target="sm_120a",
        compiler_pipeline="IntentIR->GPU->NVVM->cubin",
    )


def _variant(
    name: str,
    role: str,
    *,
    contract: RuntimeContract | None,
    fallback: str | None,
) -> VariantRecord:
    return VariantRecord(
        variant_id=name,
        role=role,
        schedule={"tile": 128},
        artifact=_artifact(name),
        launch_plan=ExecutableLaunchPlan(
            steps=(
                DeviceLaunchStep(
                    step_id="main",
                    binary_module=name,
                    entry_point=f"{name}_kernel",
                    grid=("ceildiv(x0, 128)", "1", "1"),
                    block=(128, 1, 1),
                    dynamic_smem_bytes=0,
                    device_argument_order=("p0", "p1", "p2"),
                ),
            ),
        ),
        runtime_contract=contract,
        compile_assumptions=(),
        fallback_variant_id=fallback,
    )


def _manifest() -> VariantManifest:
    fallback_id = "k0.fallback"
    fallback = _variant(
        fallback_id,
        "conservative_fallback",
        contract=None,
        fallback=None,
    )
    contract = RuntimeContract(
        contract_id="k0.tile128.contract",
        predicates=(
            RuntimePredicate(
                "k0.x0",
                "dimension_divisible",
                "x0",
                128,
                ("canonical.shape.x0",),
            ),
        ),
    )
    specialized = _variant(
        "k0.tile128",
        "specialized",
        contract=contract,
        fallback=fallback_id,
    )
    return VariantManifest(
        kernel_id="k0",
        case_id="case0",
        harness_binding_id="harness0",
        invocation_slots=(InvocationSlot("p0"), InvocationSlot("p1"), InvocationSlot("p2")),
        variants=(fallback, specialized),
        pre_tuned_order=(specialized.variant_id,),
        seed_variant_id=specialized.variant_id,
    )


def test_manifest_roundtrip_keeps_only_thin_invocation_slots() -> None:
    manifest = _manifest()
    restored = VariantManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert restored.to_dict()["invocation_slots"] == [
        {"name": "p0"}, {"name": "p1"}, {"name": "p2"}
    ]


def test_invocation_slot_rejects_a_second_semantic_interface() -> None:
    with pytest.raises(VariantManifestError, match="unknown fields"):
        InvocationSlot.from_dict(
            {"name": "p0", "dtype": "f32", "mode": "input", "rank": 1}
        )


def test_manifest_hard_rejects_pre_reset_schema_and_interface_field() -> None:
    payload = _manifest().to_dict()
    payload["schema_version"] = "intentir.backend.variant-manifest.v1"
    with pytest.raises(VariantManifestError, match="unsupported variant manifest"):
        VariantManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["logical_" + "abi"] = payload.pop("invocation_slots")
    with pytest.raises(VariantManifestError, match="missing fields"):
        VariantManifest.from_dict(copy.deepcopy(payload))
