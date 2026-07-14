"""SCRC Gate2: three-way source/replay/production-lowering execution."""

from .compare import compare_outputs
from .lower2c import Lower2CRunner, Lower2CSpec
from .model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    OutputAgreement,
    PairwiseAgreement,
    Tolerance,
)
from .orchestrate import Gate2Case, Gate2Runner
from .triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)

__all__ = [
    "ArgumentRole",
    "ExecutionResult",
    "Gate2Case",
    "Gate2Result",
    "Gate2Runner",
    "Gate2Status",
    "Lower2CRunner",
    "Lower2CSpec",
    "OutputAgreement",
    "PairwiseAgreement",
    "Tolerance",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
    "compare_outputs",
]
