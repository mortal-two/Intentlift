"""Compatibility facade for the CUDA lowering pipeline.

New target-aware callers should import :mod:`intentir.backend.cuda.pipeline`.
This module remains import-compatible for existing tools, tests, and archived
Gate integrations while the backend implementation is split by target.
"""

from intentir.backend.cuda.pipeline import (
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
