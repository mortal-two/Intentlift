"""Closed observer profiles recorded by every Gate1 certificate."""

from __future__ import annotations

from dataclasses import dataclass

from .cute import CUTE_FINGERPRINT_SCHEMA
from .model import Frontend
from .ptx import PTX_FINGERPRINT_SCHEMA
from .tilelang import TILELANG_FINGERPRINT_SCHEMA
from .ttir import TTIR_FINGERPRINT_SCHEMA


@dataclass(frozen=True)
class ObserverProfile:
    frontend: Frontend
    profile_id: str
    profile_version: str
    compiler_stage: str
    canonicalization_version: str
    evidence_class: str

    def to_json(self) -> dict[str, str]:
        return {
            "frontend": self.frontend.value,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "compiler_stage": self.compiler_stage,
            "canonicalization_version": self.canonicalization_version,
            "evidence_class": self.evidence_class,
        }


TRITON_OBSERVER_PROFILE = ObserverProfile(
    Frontend.TRITON,
    "triton_ttir_structural",
    TTIR_FINGERPRINT_SCHEMA,
    "ttir",
    TTIR_FINGERPRINT_SCHEMA,
    "strong_structural",
)

CUTE_OBSERVER_PROFILE = ObserverProfile(
    Frontend.CUTE,
    "cute_mlir_ptx_topology",
    CUTE_FINGERPRINT_SCHEMA,
    "mlir+ptx",
    CUTE_FINGERPRINT_SCHEMA,
    "strong_structural",
)

CUDA_OBSERVER_PROFILE = ObserverProfile(
    Frontend.CUDA,
    "cuda_ptx_semantic_effects",
    PTX_FINGERPRINT_SCHEMA,
    "ptx",
    PTX_FINGERPRINT_SCHEMA,
    "semantic_effects_with_raw_topology_diagnostic",
)

TILELANG_OBSERVER_PROFILE = ObserverProfile(
    Frontend.TILELANG,
    "tilelang_tir_semantic_structure",
    TILELANG_FINGERPRINT_SCHEMA,
    "tir",
    TILELANG_FINGERPRINT_SCHEMA,
    "semantic_structure_with_ptx_diagnostic",
)

_PROFILES = {
    TRITON_OBSERVER_PROFILE.frontend: TRITON_OBSERVER_PROFILE,
    CUTE_OBSERVER_PROFILE.frontend: CUTE_OBSERVER_PROFILE,
    CUDA_OBSERVER_PROFILE.frontend: CUDA_OBSERVER_PROFILE,
    TILELANG_OBSERVER_PROFILE.frontend: TILELANG_OBSERVER_PROFILE,
}


def observer_profile_for(frontend: Frontend | str) -> ObserverProfile:
    try:
        return _PROFILES[Frontend(frontend)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"no active Gate1 observer profile for {frontend!r}") from exc


__all__ = [
    "CUTE_OBSERVER_PROFILE",
    "CUDA_OBSERVER_PROFILE",
    "ObserverProfile",
    "TILELANG_OBSERVER_PROFILE",
    "TRITON_OBSERVER_PROFILE",
    "observer_profile_for",
]
