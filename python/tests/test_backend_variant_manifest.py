from __future__ import annotations

import pytest

from intentir.backend.abi import LogicalArgument
from intentir.backend.runtime_contract import RuntimeContract, RuntimePredicate
from intentir.backend.variant_manifest import (
    ArtifactProvenance,
    DeviceLaunchStep,
    LaunchPlan,
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


def _contract(name: str, tile: int) -> RuntimeContract:
    return RuntimeContract(
        contract_id=f"{name}.contract",
        predicates=(
            RuntimePredicate(
                f"{name}.n",
                "dimension_divisible",
                "N",
                tile,
                ("canonical.shape.n",),
            ),
        ),
    )


def _variant(
    name: str,
    role: str,
    *,
    tile: int,
    contract: RuntimeContract | None,
    fallback: str | None,
) -> VariantRecord:
    return VariantRecord(
        variant_id=name,
        role=role,
        schedule={"tile": tile, "vector_width": 4 if contract else 1},
        artifact=_artifact(name),
        launch_plan=LaunchPlan(
            steps=(
                DeviceLaunchStep(
                    step_id="main",
                    binary_module=name,
                    entry_point=f"{name}_kernel",
                    grid=("ceildiv(N, 256)", "M", "1"),
                    block=(256, 1, 1),
                    dynamic_smem_bytes=0,
                    device_argument_order=("X", "Y", "N"),
                ),
            ),
        ),
        runtime_contract=contract,
        compile_assumptions=(f"N specialized for tile {tile}",) if contract else (),
        fallback_variant_id=fallback,
    )


def _manifest() -> VariantManifest:
    fallback = _variant(
        "softmax.fallback",
        "conservative_fallback",
        tile=128,
        contract=None,
        fallback=None,
    )
    v128 = _variant(
        "softmax.tile128",
        "specialized",
        tile=128,
        contract=_contract("tile128", 128),
        fallback=fallback.variant_id,
    )
    v256 = _variant(
        "softmax.tile256",
        "specialized",
        tile=256,
        contract=_contract("tile256", 256),
        fallback=fallback.variant_id,
    )
    return VariantManifest(
        kernel_id="original8:softmax",
        case_id="M32-N1024-f16",
        binding_program_id="P4:softmax.intent.mlir",
        logical_abi=(
            LogicalArgument("X", "tensor", "input", "f16", 2, ("M", "N")),
            LogicalArgument("Y", "tensor", "output", "f16", 2, ("M", "N")),
            LogicalArgument("N", "scalar", "parameter", "i32"),
        ),
        variants=(fallback, v128, v256),
        pre_tuned_order=(v256.variant_id, v128.variant_id),
        seed_variant_id=v128.variant_id,
    )


def test_manifest_requires_same_bundle_fallback_and_pre_tuned_variants() -> None:
    manifest = _manifest()

    assert manifest.fallback.variant_id == "softmax.fallback"
    assert manifest.variant("softmax.tile256").fallback_variant_id == (
        "softmax.fallback"
    )
    assert manifest.pre_tuned_order == ("softmax.tile256", "softmax.tile128")


def test_manifest_round_trip_is_deterministic() -> None:
    manifest = _manifest()
    restored = VariantManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.to_json() == manifest.to_json()


def test_separately_persisted_generic_artifact_is_not_enough() -> None:
    manifest = _manifest()
    without_fallback = tuple(
        item for item in manifest.variants if item.role == "specialized"
    )

    with pytest.raises(VariantManifestError, match="exactly one conservative"):
        VariantManifest(
            kernel_id=manifest.kernel_id,
            case_id=manifest.case_id,
            binding_program_id=manifest.binding_program_id,
            logical_abi=manifest.logical_abi,
            variants=without_fallback,
            pre_tuned_order=manifest.pre_tuned_order,
            seed_variant_id=manifest.seed_variant_id,
        )


def test_specialized_variant_must_reference_bundle_fallback() -> None:
    manifest = _manifest()
    broken = tuple(
        _variant(
            item.variant_id,
            item.role,
            tile=int(item.schedule["tile"]),
            contract=item.runtime_contract,
            fallback="some.other.fallback" if item.role == "specialized" else None,
        )
        for item in manifest.variants
    )

    with pytest.raises(VariantManifestError, match="must reference fallback"):
        VariantManifest(
            kernel_id=manifest.kernel_id,
            case_id=manifest.case_id,
            binding_program_id=manifest.binding_program_id,
            logical_abi=manifest.logical_abi,
            variants=broken,
            pre_tuned_order=manifest.pre_tuned_order,
            seed_variant_id=manifest.seed_variant_id,
        )


def test_dispatch_order_must_cover_same_compiled_specialized_set() -> None:
    manifest = _manifest()

    with pytest.raises(VariantManifestError, match="every specialized"):
        VariantManifest(
            kernel_id=manifest.kernel_id,
            case_id=manifest.case_id,
            binding_program_id=manifest.binding_program_id,
            logical_abi=manifest.logical_abi,
            variants=manifest.variants,
            pre_tuned_order=("softmax.tile128",),
            seed_variant_id=manifest.seed_variant_id,
        )


def test_fallback_cannot_carry_contract() -> None:
    with pytest.raises(VariantManifestError, match="must not carry"):
        _variant(
            "bad.fallback",
            "conservative_fallback",
            tile=128,
            contract=_contract("bad", 128),
            fallback=None,
        )

