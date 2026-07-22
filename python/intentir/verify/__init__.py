"""SCRC verification entry point.

The active verification design is one lifecycle over dependent TypeCheck,
source-isolated Gate1 and external-Q three-edge Gate2. Historical helpers are
archived under ``intentir.verify.back`` and are intentionally not re-exported.
"""

from . import gate1
from . import gate2
from .adapters import BoundGate1Adapter, TritonGate2ExecutionAdapter
from .typecheck import (
    TypeCheckDiagnostic,
    TypeCheckErrorCode,
    TypeCheckReport,
    TypeCheckStatus,
    type_check,
)
from .scrc_runner import (
    CandidateLiftError,
    FeedbackKind,
    HarnessBindingInstance,
    LiftCallResult,
    LiftFeedback,
    MainStatus,
    SCRCGateRunner,
    SCRCInfrastructureError,
    SCRCRunDependencies,
    SCRCRunReport,
    SCRCRunRequest,
    SourceProgram,
)

__all__ = [
    "TypeCheckDiagnostic",
    "TypeCheckErrorCode",
    "TypeCheckReport",
    "TypeCheckStatus",
    "BoundGate1Adapter",
    "CandidateLiftError",
    "FeedbackKind",
    "HarnessBindingInstance",
    "LiftCallResult",
    "LiftFeedback",
    "MainStatus",
    "SCRCGateRunner",
    "SCRCInfrastructureError",
    "SCRCRunDependencies",
    "SCRCRunReport",
    "SCRCRunRequest",
    "SourceProgram",
    "TritonGate2ExecutionAdapter",
    "gate1",
    "gate2",
    "type_check",
]
