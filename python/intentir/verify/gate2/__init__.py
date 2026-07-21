"""SCRC Gate2: source-anchored replay and scalar differential verification."""

from .compare import GATE2_COMPARISON_PROTOCOL, compare_outputs, validate_j_inputs
from .inputs import (
    BoundaryExtent,
    GeneratedInputCase,
    InputCaseTemplate,
    InputDistribution,
    TensorInputSpec,
    boundary_1d_extents,
    generate_input_matrix,
    generate_tensor_input,
)
from .metrics import extract_gate1_gate2_observation, summarize_gate1_gate2
from .model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
    OutputAgreement,
    PairwiseAgreement,
)
from .orchestrate import (
    GATE2_EXECUTION_PROTOCOL,
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from .scalar import (
    SCALAR_LOWERING_PROTOCOL,
    ScalarLoweringRequest,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
    scalar_broadcast_axes,
)
from .triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)

__all__ = [
    "ArgumentRole",
    "BoundaryExtent",
    "ExecutionResult",
    "GATE2_COMPARISON_PROTOCOL",
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Result",
    "Gate2Runner",
    "Gate2Status",
    "Gate2Suite",
    "Gate2SuiteResult",
    "GeneratedInputCase",
    "InputCaseTemplate",
    "InputDistribution",
    "OutputAgreement",
    "PairwiseAgreement",
    "SCALAR_LOWERING_PROTOCOL",
    "ScalarLoweringRequest",
    "ScalarLoweringRunner",
    "ScalarLoweringSpec",
    "ScalarLoweringUnavailable",
    "TensorInputSpec",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
    "boundary_1d_extents",
    "compare_outputs",
    "extract_gate1_gate2_observation",
    "generate_input_matrix",
    "generate_tensor_input",
    "summarize_gate1_gate2",
    "scalar_broadcast_axes",
    "validate_j_inputs",
]
