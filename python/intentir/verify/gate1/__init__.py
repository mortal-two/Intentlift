"""LLM-first Gate1 public data boundary.

Execution helpers live in their focused modules.  This package intentionally
does not re-export retired construction engines or experiment matrices.
"""

from .model import (
    AccessKind,
    AliasKind,
    AliasSpec,
    Boundary,
    BoundaryAccess,
    BroadcastSpec,
    CompileCase,
    EffectKind,
    EffectSpec,
    Frontend,
    Gate1Artifact,
    Gate1Status,
    KernelInterface,
    KernelParameter,
    LocationEntry,
    LocationMap,
    OutputMapping,
    ParameterKind,
    ParameterMode,
    PortableScope,
    ProvenanceRecord,
    ProvenanceRole,
    ReductionBoundary,
    RuntimeArgument,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SchemaError,
    ScopeKind,
    SemanticArchive,
    SemanticEventKind,
    SemanticEventRef,
    SourceSpan,
    StorageMutability,
    StorageObject,
    opaque_replay_abi,
)
from .pi import RealizationWitness, RealizationWitnessError

__all__ = [
    "AccessKind", "AliasKind", "AliasSpec", "Boundary", "BoundaryAccess",
    "BroadcastSpec", "CompileCase", "EffectKind", "EffectSpec", "Frontend",
    "Gate1Artifact", "Gate1Status", "KernelInterface", "KernelParameter",
    "LocationEntry", "LocationMap", "OutputMapping", "ParameterKind",
    "ParameterMode", "PortableScope", "ProvenanceRecord", "ProvenanceRole",
    "RealizationWitness", "RealizationWitnessError", "ReductionBoundary",
    "RuntimeArgument", "ScheduleSlot", "ScheduleSlotKind", "ScheduleSpec",
    "SchemaError", "ScopeKind", "SemanticArchive", "SemanticEventKind",
    "SemanticEventRef", "SourceSpan", "StorageMutability", "StorageObject",
    "opaque_replay_abi",
]
