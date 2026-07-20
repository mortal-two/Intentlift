"""Stable, target-neutral IntentIR model and lowering contract.

Legacy ``IntentFunction`` remains the JSON-to-dialect graph model.  New backend
handoffs use ``IntentProgram`` so typed Layer B and the logical ABI cannot be
silently dropped.
"""

from .abi import LogicalABIError, LogicalArgument, LogicalInterface
from .contracts import (
    AccessContract,
    AccessMode,
    AliasContract,
    AliasRelation,
    BoundaryContract,
    BroadcastProjection,
    ConflictKind,
    ContractError,
    ContractExpr,
    DestinationBinding,
    EffectContract,
    EffectMode,
    EmptyDomainPolicy,
    IndexRole,
    NanPolicy,
    ObligationCategory,
    ObligationLedger,
    ObligationResult,
    ObligationStatus,
    ReassociationPolicy,
    ReductionContract,
    WriteCoverage,
)
from .model import (
    IntentFunction,
    IntentOp,
    IntentReturn,
    LoopScanInput,
    LoopStateBinding,
    LoopTrip,
    LoopValueType,
    Tensor,
)
from .program import IntentProgram, IntentProgramError, ScheduleHint, StrategyHint

__all__ = [
    "AccessContract", "AccessMode", "AliasContract", "AliasRelation",
    "BoundaryContract", "BroadcastProjection", "ConflictKind", "ContractError",
    "ContractExpr", "DestinationBinding", "EffectContract", "EffectMode",
    "EmptyDomainPolicy", "IndexRole", "IntentFunction", "IntentOp",
    "IntentProgram", "IntentProgramError", "IntentReturn", "LogicalABIError",
    "LogicalArgument", "LogicalInterface", "LoopScanInput", "LoopStateBinding",
    "LoopTrip", "LoopValueType", "NanPolicy", "ObligationCategory",
    "ObligationLedger", "ObligationResult", "ObligationStatus",
    "ReassociationPolicy", "ReductionContract", "ScheduleHint", "StrategyHint",
    "Tensor", "WriteCoverage",
]
