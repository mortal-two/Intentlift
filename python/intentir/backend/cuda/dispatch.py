"""Prepared host-side dispatch for CUDA variant bundles.

Compilation and measured tuning are offline inputs to this module.  A
``PreparedVariant`` already owns a loaded launch callable, its variant-specific
prelaunch work, and its logical-to-device ABI binder.  Runtime dispatch only
evaluates the closed performance-contract language, chooses a pre-tuned
variant, and falls back conservatively on every contract miss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from intentir.backend.runtime_contract import ContractMatch, RuntimeLaunchFacts
from intentir.backend.variant_manifest import (
    DeviceLaunchStep,
    VariantManifest,
    VariantRecord,
)


class CudaDispatchError(ValueError):
    """Raised when a prepared bundle or logical launch is inconsistent."""


DispatchMode = str
_MODES = frozenset({"default", "dispatch_off", "contract_off"})

PrelaunchCallable = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
ArgumentBinder = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Sequence[Any]]
]
LaunchCallable = Callable[[tuple[Any, ...], DeviceLaunchStep], Any]


def _no_prelaunch(logical_arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    del logical_arguments
    return {}


@dataclass(frozen=True)
class PreparedVariant:
    """Loaded executable hooks for one manifest variant.

    No compiler or autotuner callback is accepted by this API.  Callers must
    finish materialization, tuning, module loading, and graph construction
    before constructing the dispatcher used in a timed run.
    """

    variant_id: str
    bind_device_arguments: ArgumentBinder
    launch: LaunchCallable
    prelaunch: PrelaunchCallable = _no_prelaunch

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id.strip():
            raise CudaDispatchError("prepared variant_id must be non-empty")
        for name in ("bind_device_arguments", "launch", "prelaunch"):
            if not callable(getattr(self, name)):
                raise CudaDispatchError(f"prepared variant {name} must be callable")


@dataclass(frozen=True)
class DispatchSelection:
    mode: DispatchMode
    selected_variant_id: str
    fallback_used: bool
    reason: str
    evaluated_variant_ids: tuple[str, ...]
    eligible_variant_ids: tuple[str, ...]
    contract_matches: tuple[ContractMatch, ...]
    selection_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_variant_id": self.selected_variant_id,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "evaluated_variant_ids": list(self.evaluated_variant_ids),
            "eligible_variant_ids": list(self.eligible_variant_ids),
            "contract_matches": [match.to_dict() for match in self.contract_matches],
            "selection_time_ms": self.selection_time_ms,
        }


@dataclass(frozen=True)
class DispatchExecution:
    selection: DispatchSelection
    device_argument_counts: tuple[tuple[str, int], ...]
    prelaunch_time_ms: float
    binding_time_ms: float
    launch_results: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection.to_dict(),
            "device_argument_counts": dict(self.device_argument_counts),
            "prelaunch_time_ms": self.prelaunch_time_ms,
            "binding_time_ms": self.binding_time_ms,
            "launch_results": dict(self.launch_results),
        }


@dataclass(frozen=True)
class PreparedDispatchCall:
    """One selection/binding result reusable for a stable logical launch.

    Preparing a call evaluates contracts and performs the variant-specific
    prelaunch/binding once.  Repeated ``run`` calls only enqueue the already
    selected launch chain.  This models host dispatch caching by runtime
    signature and keeps compilation/autotuning/selection off the production
    kernel fast path.
    """

    selection: DispatchSelection
    record: VariantRecord
    prepared: PreparedVariant
    device_arguments: Mapping[str, tuple[Any, ...]]
    prelaunch_time_ms: float
    binding_time_ms: float

    def run(self) -> DispatchExecution:
        launch_results: list[tuple[str, Any]] = []
        for step in self.record.launch_plan.steps:
            launch_results.append(
                (
                    step.step_id,
                    self.prepared.launch(self.device_arguments[step.step_id], step),
                )
            )
        return DispatchExecution(
            selection=self.selection,
            device_argument_counts=tuple(
                (step.step_id, len(self.device_arguments[step.step_id]))
                for step in self.record.launch_plan.steps
            ),
            prelaunch_time_ms=self.prelaunch_time_ms,
            binding_time_ms=self.binding_time_ms,
            launch_results=tuple(launch_results),
        )


class CudaVariantDispatcher:
    """Dispatch a fully prepared same-ABI, same-A/B variant bundle."""

    def __init__(
        self,
        manifest: VariantManifest,
        prepared_variants: Mapping[str, PreparedVariant],
    ) -> None:
        if not isinstance(manifest, VariantManifest):
            raise CudaDispatchError("manifest must be a VariantManifest")
        prepared = dict(prepared_variants)
        manifest_ids = {variant.variant_id for variant in manifest.variants}
        if set(prepared) != manifest_ids:
            missing = sorted(manifest_ids - set(prepared))
            unknown = sorted(set(prepared) - manifest_ids)
            raise CudaDispatchError(
                f"prepared variants must exactly match manifest; missing={missing}, "
                f"unknown={unknown}"
            )
        for variant_id, item in prepared.items():
            if item.variant_id != variant_id:
                raise CudaDispatchError(
                    f"prepared variant key {variant_id!r} does not match payload "
                    f"{item.variant_id!r}"
                )
        self.manifest = manifest
        self._prepared = prepared
        self._logical_names = tuple(argument.name for argument in manifest.logical_abi)

    def _specialized_order(self, mode: DispatchMode) -> tuple[str, ...]:
        if mode == "default":
            return self.manifest.pre_tuned_order
        if mode == "dispatch_off":
            return (self.manifest.seed_variant_id,)
        return ()

    def select(
        self,
        facts: RuntimeLaunchFacts,
        *,
        mode: DispatchMode = "default",
    ) -> DispatchSelection:
        if mode not in _MODES:
            raise CudaDispatchError(f"unsupported dispatch mode {mode!r}")
        if not isinstance(facts, RuntimeLaunchFacts):
            raise CudaDispatchError("facts must be RuntimeLaunchFacts")
        started = time.perf_counter_ns()
        order = self._specialized_order(mode)
        matches: list[ContractMatch] = []
        eligible: list[str] = []
        for variant_id in order:
            variant = self.manifest.variant(variant_id)
            if variant.runtime_contract is None:
                raise CudaDispatchError(
                    f"specialized dispatch variant {variant_id!r} has no contract"
                )
            match = variant.runtime_contract.evaluate(facts)
            matches.append(match)
            if match.matched:
                eligible.append(variant_id)

        fallback = self.manifest.fallback
        if mode == "contract_off":
            selected = fallback
            reason = "contract-gated specialization disabled; conservative lowering"
        elif eligible:
            selected = self.manifest.variant(eligible[0])
            reason = (
                "pre-tuned first eligible variant"
                if mode == "default"
                else "dispatch disabled; fixed seed variant is contract-eligible"
            )
        else:
            selected = fallback
            reason = (
                "fixed seed variant missed its runtime contract"
                if mode == "dispatch_off"
                else "no specialized variant satisfied its runtime contract"
            )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return DispatchSelection(
            mode=mode,
            selected_variant_id=selected.variant_id,
            fallback_used=selected.role == "conservative_fallback",
            reason=reason,
            evaluated_variant_ids=order,
            eligible_variant_ids=tuple(eligible),
            contract_matches=tuple(matches),
            selection_time_ms=elapsed_ms,
        )

    def _check_logical_arguments(self, logical_arguments: Mapping[str, Any]) -> None:
        if not isinstance(logical_arguments, Mapping):
            raise CudaDispatchError("logical arguments must be a mapping")
        expected = set(self._logical_names)
        actual = set(logical_arguments)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise CudaDispatchError(
                f"logical ABI mismatch; missing={missing}, unknown={unknown}"
            )

    def prepare(
        self,
        logical_arguments: Mapping[str, Any],
        facts: RuntimeLaunchFacts,
        *,
        mode: DispatchMode = "default",
    ) -> PreparedDispatchCall:
        """Select and bind one stable logical call without launching it."""

        self._check_logical_arguments(logical_arguments)
        selection = self.select(facts, mode=mode)
        record: VariantRecord = self.manifest.variant(selection.selected_variant_id)
        prepared = self._prepared[record.variant_id]

        prelaunch_started = time.perf_counter_ns()
        produced = prepared.prelaunch(logical_arguments)
        prelaunch_ms = (time.perf_counter_ns() - prelaunch_started) / 1_000_000.0
        if produced is None:
            produced = {}
        if not isinstance(produced, Mapping):
            raise CudaDispatchError(
                f"prelaunch for {record.variant_id!r} must return a mapping or None"
            )

        binding_started = time.perf_counter_ns()
        raw_bindings = prepared.bind_device_arguments(
            logical_arguments, produced
        )
        if not isinstance(raw_bindings, Mapping):
            raise CudaDispatchError(
                f"device ABI binder for {record.variant_id!r} must return a step mapping"
            )
        expected_steps = {step.step_id for step in record.launch_plan.steps}
        if set(raw_bindings) != expected_steps:
            missing = sorted(expected_steps - set(raw_bindings))
            unknown = sorted(set(raw_bindings) - expected_steps)
            raise CudaDispatchError(
                f"device ABI binder for {record.variant_id!r} has step mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        device_arguments: dict[str, tuple[Any, ...]] = {}
        for step in record.launch_plan.steps:
            raw_arguments = raw_bindings[step.step_id]
            if isinstance(raw_arguments, (str, bytes)) or not isinstance(
                raw_arguments, Sequence
            ):
                raise CudaDispatchError(
                    f"device ABI binder step {step.step_id!r} must be a sequence"
                )
            arguments = tuple(raw_arguments)
            expected_count = len(step.device_argument_order)
            if len(arguments) != expected_count:
                raise CudaDispatchError(
                    f"device ABI binder step {step.step_id!r} returned "
                    f"{len(arguments)} arguments, expected {expected_count}"
                )
            device_arguments[step.step_id] = arguments
        binding_ms = (time.perf_counter_ns() - binding_started) / 1_000_000.0
        return PreparedDispatchCall(
            selection=selection,
            record=record,
            prepared=prepared,
            device_arguments=device_arguments,
            prelaunch_time_ms=prelaunch_ms,
            binding_time_ms=binding_ms,
        )

    def run(
        self,
        logical_arguments: Mapping[str, Any],
        facts: RuntimeLaunchFacts,
        *,
        mode: DispatchMode = "default",
    ) -> DispatchExecution:
        """Prepare and enqueue one call; use ``prepare`` for repeated launches."""

        return self.prepare(logical_arguments, facts, mode=mode).run()


__all__ = [
    "CudaDispatchError",
    "CudaVariantDispatcher",
    "DispatchExecution",
    "DispatchSelection",
    "PreparedDispatchCall",
    "PreparedVariant",
]
