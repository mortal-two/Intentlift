"""Target-neutral IntentIR Layer-A operation model.

Canonical J=A+B is the single ``SemanticArchive`` object used by Lift, Gate2
and target backends; it is intentionally not wrapped or re-serialized here.
"""

from .model import (
    EffectIfArm,
    IntentFunction,
    IntentOp,
    IntentReturn,
    LoopScanInput,
    LoopStateBinding,
    LoopTrip,
    LoopValueType,
    Tensor,
    ValueIfArm,
)

__all__ = [
    "EffectIfArm", "IntentFunction", "IntentOp", "IntentReturn", "LoopScanInput",
    "LoopStateBinding", "LoopTrip", "LoopValueType", "Tensor",
    "ValueIfArm",
]
