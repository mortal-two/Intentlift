"""Machine-readable lowering capability registry.

The registry reports executable backend closure only.  It intentionally has no
Gate verdict, paper stage, or certification field, so verification policy
cannot silently become a lowering authorization mechanism.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CapabilityRegistryError(ValueError):
    """Raised when the persisted lowering capability registry is inconsistent."""


_LOWERING_STATUSES = {
    "automatic_executed",
    "explicit_pass_executed",
    "exact_variant_gap",
}
_SELECTION_MODES = {"automatic", "explicit_pass", "unavailable"}
_SOURCE_ALIGNMENTS = {
    "exact_capture",
    "upstream_compared_handwritten_mlir",
    "upstream_compared_semantic_subset",
    "family_only",
}
_PERFORMANCE_EVIDENCE = {
    "archived_formal",
    "runtime_parity",
    "correctness_only",
    "none",
}
_PERFORMANCE_OUTCOMES = {
    "pass",
    "below_threshold",
    "guard_pass",
    "not_measured",
}


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapabilityRegistryError(f"{context} must be an object")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapabilityRegistryError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CapabilityRegistryError(
            f"{context} must be an array of non-empty strings"
        )
    return tuple(value)


@dataclass(frozen=True)
class KernelLoweringCapability:
    ordinal: int
    kernel_id: str
    suite: str
    symbol: str
    mechanisms: tuple[str, ...]
    source_alignment: str
    lowering_status: str
    fast_path_id: str | None
    candidate_fast_path_id: str | None
    selection: str
    generic_fallback: str
    performance_evidence: str
    performance_outcome: str
    blockers: tuple[str, ...]

    @property
    def device_execution_proven(self) -> bool:
        return self.lowering_status in {
            "automatic_executed",
            "explicit_pass_executed",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelLoweringCapability":
        lowering = _require_mapping(value.get("lowering"), "kernel.lowering")
        performance = _require_mapping(
            value.get("performance"), "kernel.performance"
        )
        ordinal = value.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise CapabilityRegistryError("kernel.ordinal must be a positive integer")
        status = _require_string(lowering.get("status"), "lowering.status")
        if status not in _LOWERING_STATUSES:
            raise CapabilityRegistryError(f"unknown lowering status {status!r}")
        selection = _require_string(lowering.get("selection"), "lowering.selection")
        if selection not in _SELECTION_MODES:
            raise CapabilityRegistryError(f"unknown selection mode {selection!r}")
        alignment = _require_string(
            value.get("source_alignment"), "kernel.source_alignment"
        )
        if alignment not in _SOURCE_ALIGNMENTS:
            raise CapabilityRegistryError(
                f"unknown source alignment {alignment!r}"
            )
        evidence = _require_string(
            performance.get("evidence"), "performance.evidence"
        )
        outcome = _require_string(
            performance.get("outcome"), "performance.outcome"
        )
        if evidence not in _PERFORMANCE_EVIDENCE:
            raise CapabilityRegistryError(
                f"unknown performance evidence {evidence!r}"
            )
        if outcome not in _PERFORMANCE_OUTCOMES:
            raise CapabilityRegistryError(
                f"unknown performance outcome {outcome!r}"
            )
        fast_path = lowering.get("fast_path_id")
        if fast_path is not None:
            fast_path = _require_string(fast_path, "lowering.fast_path_id")
        candidate = lowering.get("candidate_fast_path_id")
        if candidate is not None:
            candidate = _require_string(
                candidate, "lowering.candidate_fast_path_id"
            )
        if status == "exact_variant_gap":
            if fast_path is not None:
                raise CapabilityRegistryError(
                    "exact_variant_gap must not claim a selected fast path"
                )
        return cls(
            ordinal=ordinal,
            kernel_id=_require_string(value.get("kernel_id"), "kernel.kernel_id"),
            suite=_require_string(value.get("suite"), "kernel.suite"),
            symbol=_require_string(value.get("symbol"), "kernel.symbol"),
            mechanisms=_require_string_list(
                value.get("mechanisms"), "kernel.mechanisms"
            ),
            source_alignment=alignment,
            lowering_status=status,
            fast_path_id=fast_path,
            candidate_fast_path_id=candidate,
            selection=selection,
            generic_fallback=_require_string(
                lowering.get("generic_fallback"), "lowering.generic_fallback"
            ),
            performance_evidence=evidence,
            performance_outcome=outcome,
            blockers=_require_string_list(
                value.get("blockers", []), "kernel.blockers"
            ),
        )


@dataclass(frozen=True)
class HighPerformanceCapabilityRegistry:
    schema_version: str
    mechanisms: frozenset[str]
    kernels: tuple[KernelLoweringCapability, ...]
    scenario_29: Mapping[str, Any]
    declared_rollup: Mapping[str, Any]

    @property
    def by_id(self) -> dict[str, KernelLoweringCapability]:
        return {kernel.kernel_id: kernel for kernel in self.kernels}

    def summary(self) -> dict[str, Any]:
        performance = {
            "archived_formal_pass": sum(
                kernel.performance_evidence == "archived_formal"
                and kernel.performance_outcome == "pass"
                for kernel in self.kernels
            ),
            "archived_formal_below_threshold": sum(
                kernel.performance_evidence == "archived_formal"
                and kernel.performance_outcome == "below_threshold"
                for kernel in self.kernels
            ),
            "runtime_parity_guard": sum(
                kernel.performance_evidence == "runtime_parity"
                and kernel.performance_outcome == "guard_pass"
                for kernel in self.kernels
            ),
            "correctness_only": sum(
                kernel.performance_evidence == "correctness_only"
                for kernel in self.kernels
            ),
            "none": sum(
                kernel.performance_evidence == "none" for kernel in self.kernels
            ),
        }
        return {
            "exact_capture_alignment": sum(
                kernel.source_alignment == "exact_capture"
                for kernel in self.kernels
            ),
            "device_execution_proven": sum(
                kernel.device_execution_proven for kernel in self.kernels
            ),
            "automatic_selection_executed": sum(
                kernel.lowering_status == "automatic_executed"
                for kernel in self.kernels
            ),
            "explicit_pass_execution": sum(
                kernel.lowering_status == "explicit_pass_executed"
                for kernel in self.kernels
            ),
            "exact_variant_gaps": sum(
                kernel.lowering_status == "exact_variant_gap"
                for kernel in self.kernels
            ),
            "performance": performance,
        }


def _default_registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "spec"
        / "lowering"
        / "high-performance-kernels.json"
    )


def load_high_performance_registry(
    path: str | Path | None = None,
) -> HighPerformanceCapabilityRegistry:
    registry_path = Path(path) if path is not None else _default_registry_path()
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityRegistryError(
            f"cannot read capability registry {registry_path}: {exc}"
        ) from exc
    root = _require_mapping(raw, "registry")
    schema = _require_string(root.get("schema_version"), "registry.schema_version")
    if schema != "intentir.lowering.high-performance.v1":
        raise CapabilityRegistryError(f"unsupported registry schema {schema!r}")
    mechanisms_raw = _require_mapping(root.get("mechanisms"), "registry.mechanisms")
    mechanism_ids = frozenset(
        _require_string(key, "mechanism id") for key in mechanisms_raw
    )
    kernels_raw = root.get("kernels")
    if not isinstance(kernels_raw, list):
        raise CapabilityRegistryError("registry.kernels must be an array")
    kernels = tuple(
        KernelLoweringCapability.from_dict(
            _require_mapping(item, f"registry.kernels[{index}]")
        )
        for index, item in enumerate(kernels_raw)
    )
    ordinals = [kernel.ordinal for kernel in kernels]
    if ordinals != list(range(1, 29)):
        raise CapabilityRegistryError(
            "high-performance kernel ordinals must be exactly 1..28"
        )
    ids = [kernel.kernel_id for kernel in kernels]
    if len(set(ids)) != len(ids):
        raise CapabilityRegistryError("kernel ids must be unique")
    symbols = [(kernel.suite, kernel.symbol) for kernel in kernels]
    if len(set(symbols)) != len(symbols):
        raise CapabilityRegistryError("suite/symbol identities must be unique")
    unknown_mechanisms = sorted(
        {
            mechanism
            for kernel in kernels
            for mechanism in kernel.mechanisms
            if mechanism not in mechanism_ids
        }
    )
    if unknown_mechanisms:
        raise CapabilityRegistryError(
            f"kernels reference unknown mechanisms: {unknown_mechanisms}"
        )
    counting = _require_mapping(root.get("counting"), "registry.counting")
    if counting.get("direct_jit_kernels") != 28 or counting.get(
        "total_scenarios"
    ) != 29:
        raise CapabilityRegistryError(
            "registry counting must preserve 28 direct kernels / 29 scenarios"
        )
    scenario = _require_mapping(root.get("scenario_29"), "registry.scenario_29")
    if scenario.get("counted_as_direct_kernel") is not False:
        raise CapabilityRegistryError("scenario 29 must not count as a direct JIT")
    declared = _require_mapping(root.get("rollup"), "registry.rollup")
    result = HighPerformanceCapabilityRegistry(
        schema_version=schema,
        mechanisms=mechanism_ids,
        kernels=kernels,
        scenario_29=dict(scenario),
        declared_rollup=dict(declared),
    )
    if result.summary() != dict(declared):
        raise CapabilityRegistryError(
            "declared registry rollup does not match kernel entries"
        )
    return result


__all__ = [
    "CapabilityRegistryError",
    "HighPerformanceCapabilityRegistry",
    "KernelLoweringCapability",
    "load_high_performance_registry",
]
