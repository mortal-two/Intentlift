"""NVIDIA CUDA target backend.

The package is intentionally lazy so importing a compatibility module such as
``intentir.backend.compile`` can alias the canonical CUDA implementation
without first importing the complete compile/selection/runtime stack.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "LoweredProgram",
    "execute_cuda_artifact",
    "lower_function_to_cuda",
    "lower_to_cuda",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("intentir.backend.cuda.pipeline"), name)
