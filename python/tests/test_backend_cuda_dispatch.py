from __future__ import annotations

import pytest

from intentir.backend.abi import LogicalArgument
from intentir.backend.cuda_dispatch import (
    CudaDispatchError,
    CudaVariantDispatcher,
    PreparedVariant,
)
from intentir.backend.runtime_contract import (
    RuntimeContract,
    RuntimeLaunchFacts,
    RuntimePredicate,
)
from intentir.backend.variant_manifest import (
    ArtifactProvenance,
    DeviceLaunchStep,
    LaunchPlan,
    VariantManifest,
    VariantRecord,
)


def _artifact(name: str, step_count: int) -> ArtifactProvenance:
    modules = tuple(f"{name}.module{index}" for index in range(step_count))
    entries = tuple(f"{name}_entry{index}" for index in range(step_count))
    return ArtifactProvenance(
        intent_mlir_paths=(f"{name}.mlir",),
        binary_paths=tuple(f"{name}.{index}.cubin" for index in range(step_count)),
        binary_modules=modules,
        entry_points=entries,
        target="sm_120a",
        compiler_pipeline="IntentIR->GPU->NVVM->cubin",
    )


def _contract(name: str, divisor: int) -> RuntimeContract:
    return RuntimeContract(
        contract_id=f"{name}.contract",
        predicates=(
            RuntimePredicate(
                f"{name}.n",
                "dimension_divisible",
                "N",
                divisor,
                ("canonical.shape.n",),
            ),
            RuntimePredicate(
                f"{name}.stride",
                "stride_eq",
                "X:1",
                1,
                ("canonical.layout.x",),
            ),
        ),
    )


def _record(
    name: str,
    role: str,
    divisor: int,
    *,
    fallback: str | None,
    prelaunch_steps=(),
    step_count: int = 1,
) -> VariantRecord:
    specialized = role == "specialized"
    artifact = _artifact(name, step_count)
    return VariantRecord(
        variant_id=name,
        role=role,
        schedule={"tile": divisor, "vector_width": 4 if specialized else 1},
        artifact=artifact,
        launch_plan=LaunchPlan(
            steps=tuple(
                DeviceLaunchStep(
                    step_id=f"main{index}",
                    binary_module=artifact.binary_modules[index],
                    entry_point=artifact.entry_points[index],
                    grid=("ceildiv(N,tile)", "M", "1"),
                    block=(256, 1, 1),
                    dynamic_smem_bytes=0,
                    device_argument_order=("X", "Y", "N"),
                )
                for index in range(step_count)
            ),
            prelaunch_steps=tuple(prelaunch_steps),
        ),
        runtime_contract=_contract(name, divisor) if specialized else None,
        compile_assumptions=(f"tile={divisor}",) if specialized else (),
        fallback_variant_id=fallback,
    )


def _manifest() -> VariantManifest:
    fallback = _record(
        "softmax.fallback",
        "conservative_fallback",
        64,
        fallback=None,
        step_count=2,
    )
    v128 = _record(
        "softmax.v128", "specialized", 128, fallback=fallback.variant_id
    )
    v256 = _record(
        "softmax.v256",
        "specialized",
        256,
        fallback=fallback.variant_id,
        prelaunch_steps=("prepare_workspace",),
    )
    return VariantManifest(
        kernel_id="original8:softmax",
        case_id="dynamic-N",
        binding_program_id="P4:softmax",
        logical_abi=(
            LogicalArgument("X", "tensor", "input", "f16", 2, ("M", "N")),
            LogicalArgument("Y", "tensor", "output", "f16", 2, ("M", "N")),
            LogicalArgument("N", "scalar", "parameter", "i32"),
        ),
        variants=(fallback, v128, v256),
        pre_tuned_order=(v256.variant_id, v128.variant_id),
        seed_variant_id=v256.variant_id,
    )


def _facts(n: int, *, stride: int = 1) -> RuntimeLaunchFacts:
    return RuntimeLaunchFacts(
        dimensions={"M": 32, "N": n},
        strides={"X": (n, stride), "Y": (n, 1)},
        pointer_addresses={"X": 0x1000, "Y": 0x2000},
        target_capabilities=frozenset({"sm_120a"}),
    )


def _prepared(manifest: VariantManifest, events: list[str]):
    result = {}
    for record in manifest.variants:
        variant_id = record.variant_id

        def prelaunch(logical, *, variant_id=variant_id):
            events.append(f"prelaunch:{variant_id}")
            return {"workspace": f"workspace:{variant_id}"}

        def bind(logical, produced, *, variant_id=variant_id, record=record):
            events.append(f"bind:{variant_id}")
            assert produced["workspace"] == f"workspace:{variant_id}"
            return {
                step.step_id: (logical["X"], logical["Y"], logical["N"])
                for step in record.launch_plan.steps
            }

        def launch(arguments, step, *, variant_id=variant_id):
            events.append(f"launch:{variant_id}:{step.step_id}")
            return {
                "variant_id": variant_id,
                "arguments": arguments,
                "step_id": step.step_id,
                "block": step.block,
            }

        result[variant_id] = PreparedVariant(
            variant_id=variant_id,
            prelaunch=prelaunch,
            bind_device_arguments=bind,
            launch=launch,
        )
    return result


def _dispatcher(events: list[str]) -> CudaVariantDispatcher:
    manifest = _manifest()
    return CudaVariantDispatcher(manifest, _prepared(manifest, events))


def test_default_selects_first_contract_eligible_pre_tuned_variant() -> None:
    dispatcher = _dispatcher([])

    selection = dispatcher.select(_facts(1024))

    assert selection.selected_variant_id == "softmax.v256"
    assert selection.eligible_variant_ids == ("softmax.v256", "softmax.v128")
    assert selection.fallback_used is False
    assert selection.selection_time_ms >= 0.0


def test_default_falls_through_order_then_to_conservative_bundle_variant() -> None:
    dispatcher = _dispatcher([])

    v128 = dispatcher.select(_facts(384))
    fallback = dispatcher.select(_facts(130))

    assert v128.selected_variant_id == "softmax.v128"
    assert fallback.selected_variant_id == "softmax.fallback"
    assert fallback.fallback_used is True


def test_dispatch_off_uses_fixed_seed_and_never_adapts_to_other_variant() -> None:
    dispatcher = _dispatcher([])

    # v128 is legal for N=384, but fixed seed v256 is not.  Safety wins and the
    # conservative fallback runs; dispatch_off must not adaptively pick v128.
    selection = dispatcher.select(_facts(384), mode="dispatch_off")

    assert selection.evaluated_variant_ids == ("softmax.v256",)
    assert selection.eligible_variant_ids == ()
    assert selection.selected_variant_id == "softmax.fallback"
    assert selection.fallback_used is True


def test_contract_off_forces_conservative_lowering_without_mutating_bundle() -> None:
    dispatcher = _dispatcher([])

    selection = dispatcher.select(_facts(1024), mode="contract_off")

    assert selection.selected_variant_id == "softmax.fallback"
    assert selection.evaluated_variant_ids == ()
    assert selection.contract_matches == ()
    assert set(dispatcher._prepared) == {
        "softmax.fallback",
        "softmax.v128",
        "softmax.v256",
    }


def test_run_uses_selected_variant_prelaunch_and_device_abi_mapping() -> None:
    events: list[str] = []
    dispatcher = _dispatcher(events)

    execution = dispatcher.run(
        {"X": "xptr", "Y": "yptr", "N": 1024}, _facts(1024)
    )

    assert events == [
        "prelaunch:softmax.v256",
        "bind:softmax.v256",
        "launch:softmax.v256:main0",
    ]
    assert dict(execution.launch_results)["main0"]["variant_id"] == "softmax.v256"
    assert dict(execution.device_argument_counts) == {"main0": 3}


def test_prepared_call_reuses_selection_and_binding_for_repeated_launches() -> None:
    events: list[str] = []
    dispatcher = _dispatcher(events)

    prepared = dispatcher.prepare(
        {"X": "xptr", "Y": "yptr", "N": 1024}, _facts(1024)
    )
    assert events == ["prelaunch:softmax.v256", "bind:softmax.v256"]

    first = prepared.run()
    second = prepared.run()

    assert events == [
        "prelaunch:softmax.v256",
        "bind:softmax.v256",
        "launch:softmax.v256:main0",
        "launch:softmax.v256:main0",
    ]
    assert first.selection == second.selection == prepared.selection


def test_conservative_fallback_can_be_a_real_multi_kernel_chain() -> None:
    events: list[str] = []
    dispatcher = _dispatcher(events)

    execution = dispatcher.run(
        {"X": "xptr", "Y": "yptr", "N": 130}, _facts(130)
    )

    assert execution.selection.selected_variant_id == "softmax.fallback"
    assert events == [
        "prelaunch:softmax.fallback",
        "bind:softmax.fallback",
        "launch:softmax.fallback:main0",
        "launch:softmax.fallback:main1",
    ]
    assert dict(execution.device_argument_counts) == {"main0": 3, "main1": 3}


def test_selection_is_side_effect_free_and_does_not_run_prelaunch() -> None:
    events: list[str] = []
    dispatcher = _dispatcher(events)

    dispatcher.select(_facts(1024))

    assert events == []


def test_logical_abi_is_checked_before_any_variant_side_effect() -> None:
    events: list[str] = []
    dispatcher = _dispatcher(events)

    with pytest.raises(CudaDispatchError, match="logical ABI mismatch"):
        dispatcher.run({"X": "xptr", "N": 1024}, _facts(1024))
    assert events == []


def test_dispatcher_requires_every_manifest_variant_prepared_up_front() -> None:
    manifest = _manifest()
    prepared = _prepared(manifest, [])
    prepared.pop("softmax.fallback")

    with pytest.raises(CudaDispatchError, match="exactly match manifest"):
        CudaVariantDispatcher(manifest, prepared)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(CudaDispatchError, match="unsupported dispatch mode"):
        _dispatcher([]).select(_facts(1024), mode="gate1")

