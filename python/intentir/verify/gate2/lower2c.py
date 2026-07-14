"""IntentIR production-lowering execution leg for Gate2.

The project currently lowers the Intent dialect through linalg/loops to LLVM
and executes it with ``mlir-runner``.  This module exposes that existing path as
the initial ``lower2c`` Gate2 prototype; it deliberately does not evaluate JSON
with the archived NumPy interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from intentir.backend.cpu import run_on_cpu
from intentir.ir.model import IntentFunction

from .model import ExecutionResult


@dataclass(frozen=True)
class Lower2CSpec:
    intent: IntentFunction
    dims: Mapping[str, int]
    input_bindings: Mapping[str, str]
    output_key: str

    def __post_init__(self) -> None:
        declared = {tensor.name for tensor in self.intent.inputs}
        missing = sorted(declared - set(self.input_bindings))
        unknown = sorted(set(self.input_bindings) - declared)
        if missing or unknown:
            raise ValueError(
                f"lower2c input bindings mismatch; missing={missing}, unknown={unknown}"
            )
        if len(self.intent.outputs) != 1:
            raise ValueError("lower2c prototype currently supports one output")


class Lower2CRunner:
    def run(self, spec: Lower2CSpec, values: Mapping[str, Any]) -> ExecutionResult:
        try:
            inputs = [
                np.asarray(values[spec.input_bindings[tensor.name]])
                for tensor in spec.intent.inputs
            ]
            output = run_on_cpu(spec.intent, dict(spec.dims), inputs)
            return ExecutionResult(
                "lowering",
                True,
                {spec.output_key: np.asarray(output)},
                metadata={
                    "path": "IntentIR->intent MLIR->linalg->LLVM->mlir-runner",
                    "function": spec.intent.name,
                },
            )
        except (NotImplementedError, ValueError) as exc:
            return ExecutionResult(
                "lowering",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
            )
        except Exception as exc:
            return ExecutionResult(
                "lowering", False, error=f"{type(exc).__name__}: {exc}"
            )


__all__ = ["Lower2CRunner", "Lower2CSpec"]
