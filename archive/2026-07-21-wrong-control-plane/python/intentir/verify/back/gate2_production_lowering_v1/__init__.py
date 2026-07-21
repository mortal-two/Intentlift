"""Archived three-way source/replay/production-lowering Gate2 package."""

from intentir.verify.gate2.compare import GATE2_COMPARISON_PROTOCOL, compare_outputs
from .cuda_lowering import (
    CUDA_LOWERING_PROTOCOL,
    CudaLoweringArtifact,
    CudaLoweringOptions,
    CudaLoweringRunner,
    CudaLoweringSpec,
    GateIntentBundle,
)
from intentir.verify.gate2.inputs import (
    BoundaryExtent,
    GeneratedInputCase,
    InputCaseTemplate,
    InputDistribution,
    TensorInputSpec,
    boundary_1d_extents,
    generate_input_matrix,
    generate_tensor_input,
)
from .lower2c import (
    LOWER2C_PROTOCOL,
    Lower2CRunner,
    Lower2CSpec,
    boundary_broadcast_axes,
)
from .metrics import extract_gate1_gate2_observation, summarize_gate1_gate2
from .model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
    OutputAgreement,
    PairwiseAgreement,
    Tolerance,
)
from .orchestrate import Gate2Case, Gate2Runner, Gate2Suite
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)

__all__ = [
    "ArgumentRole",
    "BoundaryExtent",
    "CUDA_LOWERING_PROTOCOL",
    "CudaLoweringArtifact",
    "CudaLoweringOptions",
    "CudaLoweringRunner",
    "CudaLoweringSpec",
    "ExecutionResult",
    "Gate2Case",
    "GATE2_COMPARISON_PROTOCOL",
    "Gate2Result",
    "Gate2Runner",
    "Gate2Status",
    "Gate2Suite",
    "Gate2SuiteResult",
    "GeneratedInputCase",
    "GateIntentBundle",
    "InputCaseTemplate",
    "InputDistribution",
    "LOWER2C_PROTOCOL",
    "Lower2CRunner",
    "Lower2CSpec",
    "OutputAgreement",
    "PairwiseAgreement",
    "Tolerance",
    "TensorInputSpec",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
    "boundary_1d_extents",
    "boundary_broadcast_axes",
    "compare_outputs",
    "extract_gate1_gate2_observation",
    "generate_input_matrix",
    "generate_tensor_input",
    "summarize_gate1_gate2",
]
