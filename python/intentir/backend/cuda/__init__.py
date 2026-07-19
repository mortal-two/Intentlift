"""NVIDIA CUDA target backend.

The target namespace is the canonical home for CUDA compilation, selection,
dispatch, and execution APIs.  Flat ``intentir.backend.*`` imports remain as
compatibility entry points during the staged migration.
"""

from .pipeline import (
    LoweredProgram,
    execute_cuda_artifact,
    lower_function_to_cuda,
    lower_to_cuda,
)

__all__ = [
    "LoweredProgram",
    "execute_cuda_artifact",
    "lower_function_to_cuda",
    "lower_to_cuda",
]
