"""Manifest for one kernel's conservative and specialized CUDA variants.

The manifest is the boundary between offline materialization/tuning and the
production host dispatcher.  Every variant keeps the same positional harness
binding and J, while schedule choices, device ABI mapping, prelaunch
steps, and compiled artifacts may differ.  Exactly one conservative fallback
is mandatory in the same bundle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .runtime_contract import RuntimeContract


class VariantManifestError(ValueError):
    """Raised when a multi-variant CUDA bundle is structurally invalid."""


_ROLES = frozenset({"conservative_fallback", "specialized"})


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VariantManifestError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VariantManifestError(f"{field_name} must be an integer >= {minimum}")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise VariantManifestError(f"{context} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise VariantManifestError(f"{context} missing fields: {missing}")
    if unknown:
        raise VariantManifestError(f"{context} has unknown fields: {unknown}")


def _schedule(value: Mapping[str, Any]) -> dict[str, bool | int | float | str]:
    if not isinstance(value, Mapping) or not value:
        raise VariantManifestError("variant schedule must be a non-empty object")
    result: dict[str, bool | int | float | str] = {}
    for raw_name, raw_item in sorted(value.items()):
        name = _identifier(raw_name, field_name="schedule parameter")
        if isinstance(raw_item, str):
            item: bool | int | float | str = _identifier(
                raw_item, field_name=f"schedule parameter {name}"
            )
        elif isinstance(raw_item, (bool, int, float)):
            item = raw_item
        else:
            raise VariantManifestError(
                f"schedule parameter {name!r} must be a JSON scalar"
            )
        result[name] = item
    return result


@dataclass(frozen=True)
class InvocationSlot:
    """Thin opaque harness name in positional order; carries no semantics."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _identifier(self.name, field_name="invocation slot")
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvocationSlot":
        _strict_keys(value, required={"name"}, context="invocation slot")
        return cls(value["name"])


@dataclass(frozen=True)
class DeviceLaunchStep:
    step_id: str
    binary_module: str
    entry_point: str
    grid: tuple[str, str, str]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int
    device_argument_order: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, field_name="step_id"))
        object.__setattr__(self, "binary_module", _identifier(self.binary_module, field_name="binary_module"))
        object.__setattr__(self, "entry_point", _identifier(self.entry_point, field_name="entry_point"))
        if len(self.grid) != 3:
            raise VariantManifestError("launch grid must contain three expressions")
        object.__setattr__(
            self,
            "grid",
            tuple(_identifier(item, field_name="grid expression") for item in self.grid),
        )
        if len(self.block) != 3:
            raise VariantManifestError("launch block must contain three dimensions")
        block = tuple(
            _integer(item, field_name="block dimension", minimum=1)
            for item in self.block
        )
        if block[0] * block[1] * block[2] > 1024:
            raise VariantManifestError("launch block exceeds 1024 threads")
        object.__setattr__(self, "block", block)
        object.__setattr__(
            self,
            "dynamic_smem_bytes",
            _integer(self.dynamic_smem_bytes, field_name="dynamic_smem_bytes"),
        )
        arguments = tuple(
            _identifier(item, field_name="device argument")
            for item in self.device_argument_order
        )
        if len(set(arguments)) != len(arguments):
            raise VariantManifestError("device_argument_order must not contain duplicates")
        object.__setattr__(self, "device_argument_order", arguments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "binary_module": self.binary_module,
            "entry_point": self.entry_point,
            "grid": list(self.grid),
            "block": list(self.block),
            "dynamic_smem_bytes": self.dynamic_smem_bytes,
            "device_argument_order": list(self.device_argument_order),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeviceLaunchStep":
        _strict_keys(
            value,
            required={
                "step_id",
                "binary_module",
                "entry_point",
                "grid",
                "block",
                "dynamic_smem_bytes",
                "device_argument_order",
            },
            context="device launch step",
        )
        return cls(
            step_id=value["step_id"],
            binary_module=value["binary_module"],
            entry_point=value["entry_point"],
            grid=tuple(value["grid"]),
            block=tuple(value["block"]),
            dynamic_smem_bytes=value["dynamic_smem_bytes"],
            device_argument_order=tuple(value["device_argument_order"]),
        )


@dataclass(frozen=True)
class ExecutableLaunchPlan:
    steps: tuple[DeviceLaunchStep, ...]
    prelaunch_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise VariantManifestError("launch plan requires at least one device step")
        identifiers = [step.step_id for step in self.steps]
        if len(set(identifiers)) != len(identifiers):
            raise VariantManifestError("device launch step ids must be unique")
        object.__setattr__(
            self,
            "prelaunch_steps",
            tuple(
                _identifier(item, field_name="prelaunch step")
                for item in self.prelaunch_steps
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "prelaunch_steps": list(self.prelaunch_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutableLaunchPlan":
        _strict_keys(
            value,
            required={"steps", "prelaunch_steps"},
            context="launch plan",
        )
        return cls(
            steps=tuple(DeviceLaunchStep.from_dict(item) for item in value["steps"]),
            prelaunch_steps=tuple(value["prelaunch_steps"]),
        )


@dataclass(frozen=True)
class VariantRecord:
    variant_id: str
    role: str
    schedule: Mapping[str, bool | int | float | str]
    launch_plan: ExecutableLaunchPlan
    runtime_contract: RuntimeContract | None
    compile_assumptions: tuple[str, ...]
    fallback_variant_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _identifier(self.variant_id, field_name="variant_id"))
        if self.role not in _ROLES:
            raise VariantManifestError(f"unsupported variant role {self.role!r}")
        object.__setattr__(self, "schedule", _schedule(self.schedule))
        assumptions = tuple(
            _identifier(item, field_name="compile assumption")
            for item in self.compile_assumptions
        )
        object.__setattr__(self, "compile_assumptions", assumptions)
        if self.role == "conservative_fallback":
            if self.runtime_contract is not None:
                raise VariantManifestError(
                    "conservative fallback must not carry a specialized runtime contract"
                )
            if self.fallback_variant_id is not None:
                raise VariantManifestError(
                    "conservative fallback must not reference another fallback"
                )
        else:
            if self.runtime_contract is None:
                raise VariantManifestError(
                    "specialized variant requires a runtime contract"
                )
            object.__setattr__(
                self,
                "fallback_variant_id",
                _identifier(
                    self.fallback_variant_id, field_name="fallback_variant_id"
                ),
            )
    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "role": self.role,
            "schedule": dict(self.schedule),
            "launch_plan": self.launch_plan.to_dict(),
            "runtime_contract": (
                self.runtime_contract.to_dict()
                if self.runtime_contract is not None
                else None
            ),
            "compile_assumptions": list(self.compile_assumptions),
            "fallback_variant_id": self.fallback_variant_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariantRecord":
        _strict_keys(
            value,
            required={
                "variant_id",
                "role",
                "schedule",
                "launch_plan",
                "runtime_contract",
                "compile_assumptions",
                "fallback_variant_id",
            },
            context="variant record",
        )
        runtime_contract = value["runtime_contract"]
        return cls(
            variant_id=value["variant_id"],
            role=value["role"],
            schedule=value["schedule"],
            launch_plan=ExecutableLaunchPlan.from_dict(value["launch_plan"]),
            runtime_contract=(
                RuntimeContract.from_dict(runtime_contract)
                if runtime_contract is not None
                else None
            ),
            compile_assumptions=tuple(value["compile_assumptions"]),
            fallback_variant_id=value["fallback_variant_id"],
        )


@dataclass(frozen=True)
class VariantManifest:
    kernel_id: str
    case_id: str
    harness_binding_id: str
    invocation_slots: tuple[InvocationSlot, ...]
    variants: tuple[VariantRecord, ...]
    pre_tuned_order: tuple[str, ...]
    seed_variant_id: str
    schema_version: str = "intentir.backend.variant-manifest.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "intentir.backend.variant-manifest.v2":
            raise VariantManifestError(
                f"unsupported variant manifest schema {self.schema_version!r}"
            )
        object.__setattr__(self, "kernel_id", _identifier(self.kernel_id, field_name="kernel_id"))
        object.__setattr__(self, "case_id", _identifier(self.case_id, field_name="case_id"))
        object.__setattr__(
            self,
            "harness_binding_id",
            _identifier(self.harness_binding_id, field_name="harness_binding_id"),
        )
        slots = tuple(
            item if isinstance(item, InvocationSlot) else InvocationSlot.from_dict(item)
            for item in self.invocation_slots
        )
        object.__setattr__(self, "invocation_slots", slots)
        if not slots:
            raise VariantManifestError("invocation_slots must not be empty")
        argument_names = [argument.name for argument in slots]
        if len(set(argument_names)) != len(argument_names):
            raise VariantManifestError("invocation slot names must be unique")
        if len(self.variants) < 2:
            raise VariantManifestError(
                "variant bundle requires fallback plus specialized variant"
            )
        by_id = {variant.variant_id: variant for variant in self.variants}
        if len(by_id) != len(self.variants):
            raise VariantManifestError("variant ids must be unique")
        fallbacks = [
            variant
            for variant in self.variants
            if variant.role == "conservative_fallback"
        ]
        if len(fallbacks) != 1:
            raise VariantManifestError(
                "variant bundle requires exactly one conservative fallback"
            )
        fallback_id = fallbacks[0].variant_id
        specialized = [
            variant for variant in self.variants if variant.role == "specialized"
        ]
        if not specialized:
            raise VariantManifestError("variant bundle requires specialized variants")
        for variant in specialized:
            if variant.fallback_variant_id != fallback_id:
                raise VariantManifestError(
                    f"specialized variant {variant.variant_id!r} must reference "
                    f"fallback {fallback_id!r}"
                )
        order = tuple(
            _identifier(item, field_name="pre_tuned_order item")
            for item in self.pre_tuned_order
        )
        object.__setattr__(self, "pre_tuned_order", order)
        specialized_ids = {variant.variant_id for variant in specialized}
        if len(order) != len(set(order)) or set(order) != specialized_ids:
            raise VariantManifestError(
                "pre_tuned_order must contain every specialized variant exactly once"
            )
        seed = _identifier(self.seed_variant_id, field_name="seed_variant_id")
        if seed not in specialized_ids:
            raise VariantManifestError("seed_variant_id must name a specialized variant")
        object.__setattr__(self, "seed_variant_id", seed)
        contract_ids = [
            variant.runtime_contract.contract_id
            for variant in specialized
            if variant.runtime_contract is not None
        ]
        if len(contract_ids) != len(set(contract_ids)):
            raise VariantManifestError("specialized runtime contract ids must be unique")

    @property
    def fallback(self) -> VariantRecord:
        return next(
            variant
            for variant in self.variants
            if variant.role == "conservative_fallback"
        )

    def variant(self, variant_id: str) -> VariantRecord:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise KeyError(variant_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kernel_id": self.kernel_id,
            "case_id": self.case_id,
            "harness_binding_id": self.harness_binding_id,
            "invocation_slots": [item.to_dict() for item in self.invocation_slots],
            "variants": [variant.to_dict() for variant in self.variants],
            "pre_tuned_order": list(self.pre_tuned_order),
            "seed_variant_id": self.seed_variant_id,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariantManifest":
        _strict_keys(
            value,
            required={
                "schema_version",
                "kernel_id",
                "case_id",
                "harness_binding_id",
                "invocation_slots",
                "variants",
                "pre_tuned_order",
                "seed_variant_id",
            },
            context="variant manifest",
        )
        return cls(
            schema_version=value["schema_version"],
            kernel_id=value["kernel_id"],
            case_id=value["case_id"],
            harness_binding_id=value["harness_binding_id"],
            invocation_slots=tuple(
                InvocationSlot.from_dict(item) for item in value["invocation_slots"]
            ),
            variants=tuple(VariantRecord.from_dict(item) for item in value["variants"]),
            pre_tuned_order=tuple(value["pre_tuned_order"]),
            seed_variant_id=value["seed_variant_id"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "VariantManifest":
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise VariantManifestError(f"invalid variant manifest JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise VariantManifestError("variant manifest root must be an object")
        return cls.from_dict(value)


__all__ = [
    "DeviceLaunchStep",
    "ExecutableLaunchPlan",
    "InvocationSlot",
    "VariantManifest",
    "VariantManifestError",
    "VariantRecord",
]
