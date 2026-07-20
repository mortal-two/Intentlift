"""Backend-neutral Gate2 adapter for deterministic scalar CPU lowering.

The active Gate2 compares an executable replay of the final accepted J with a
scalar execution of the same typed :class:`IntentProgram`.  This module owns
only the verification adapter.  It has no CUDA/ROCm imports, target selection,
or production-backend fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.backend.cpu.abi import (
    CPUABIValidationError,
    CPUBackendUnsupportedError,
)
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.ir.model import IntentOp
from intentir.ir.program import IntentProgram

from .model import ExecutionResult


SCALAR_LOWERING_PROTOCOL = "intent_program_scalar_cpu_mlir_v1"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _checked_digest(value: str, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _canonical_int_mapping(
    value: Mapping[str, int], context: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for key, item in value.items()
    ):
        raise ValueError(f"{context} must map names to non-negative integers")
    return tuple(sorted(value.items()))


def _canonical_str_mapping(
    value: Mapping[str, str], context: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and item
        for key, item in value.items()
    ):
        raise ValueError(f"{context} must map non-empty names to non-empty names")
    return tuple(sorted(value.items()))


@dataclass(frozen=True)
class ScalarLoweringSpec:
    """Digest-bound scalar execution request derived from one final Gate1 J."""

    program: IntentProgram
    final_artifact_digest: str
    gate1_j_digest: str
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.program, IntentProgram):
            raise ValueError("scalar lowering requires a typed IntentProgram")
        object.__setattr__(
            self,
            "final_artifact_digest",
            _checked_digest(self.final_artifact_digest, "final artifact digest"),
        )
        object.__setattr__(
            self,
            "gate1_j_digest",
            _checked_digest(self.gate1_j_digest, "Gate1 J digest"),
        )
        dims = dict(self.dims)
        inputs = dict(self.input_bindings)
        outputs = dict(self.output_bindings)
        object.__setattr__(self, "dims", _canonical_int_mapping(dims, "scalar dims"))
        object.__setattr__(
            self,
            "input_bindings",
            _canonical_str_mapping(inputs, "scalar input bindings"),
        )
        object.__setattr__(
            self,
            "output_bindings",
            _canonical_str_mapping(outputs, "scalar output bindings"),
        )
        expected_inputs = {item.name for item in self.program.function.inputs}
        expected_outputs = {item.name for item in self.program.function.outputs}
        if set(inputs) != expected_inputs:
            raise ValueError(
                "scalar input bindings differ from IntentProgram inputs: "
                f"missing={sorted(expected_inputs - set(inputs))}, "
                f"unknown={sorted(set(inputs) - expected_inputs)}"
            )
        if set(outputs) != expected_outputs:
            raise ValueError(
                "scalar output bindings differ from IntentProgram outputs: "
                f"missing={sorted(expected_outputs - set(outputs))}, "
                f"unknown={sorted(set(outputs) - expected_outputs)}"
            )
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("scalar output binding destinations must be unique")

    @classmethod
    def create(
        cls,
        program: IntentProgram,
        *,
        final_artifact_digest: str,
        gate1_j_digest: str,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
    ) -> "ScalarLoweringSpec":
        return cls(
            program,
            final_artifact_digest,
            gate1_j_digest,
            tuple(dims.items()),
            tuple(input_bindings.items()),
            tuple(output_bindings.items()),
        )

    @property
    def dims_map(self) -> dict[str, int]:
        return dict(self.dims)

    @property
    def input_binding_map(self) -> dict[str, str]:
        return dict(self.input_bindings)

    @property
    def output_binding_map(self) -> dict[str, str]:
        return dict(self.output_bindings)

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "final_artifact_digest": self.final_artifact_digest,
                "gate1_j_digest": self.gate1_j_digest,
                "intent_j_digest": self.program.j_digest,
                "intent_contract_digest": self.program.contract_digest,
                "dims": self.dims_map,
                "input_bindings": self.input_binding_map,
                "output_bindings": self.output_binding_map,
            }
        )


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def scalar_broadcast_axes(
    program: IntentProgram,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Translate typed semantic-axis insertion contracts for the MLIR emitter."""

    nodes = {
        op.node_id: op
        for op in _walk_ops(program.function.body)
        if op.node_id is not None
    }
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in program.boundary.broadcasts:
        op = nodes.get(projection.node_ref)
        if op is None or not isinstance(op.result, str):
            raise NotImplementedError(
                f"scalar broadcast {projection.contract_id!r} requires a "
                "single-result consumer"
            )
        if projection.operand not in op.operands:
            raise ValueError(
                f"scalar broadcast {projection.contract_id!r} names a "
                "non-consumed operand"
            )
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in projection.inserted_semantic_axes
        ):
            raise NotImplementedError(
                f"scalar broadcast {projection.contract_id!r} uses symbolic axes"
            )
        axes = tuple(int(axis) for axis in projection.inserted_semantic_axes)
        operands = result.setdefault(op.result, {})
        previous = operands.get(projection.operand)
        if previous is not None and previous != axes:
            raise ValueError(
                f"scalar broadcast {projection.contract_id!r} conflicts with "
                "another projection"
            )
        operands[projection.operand] = axes
    return result


class ScalarLoweringRunner:
    """Execute typed Layer-A semantics through the scalar CPU MLIR pipeline."""

    def run(
        self, spec: ScalarLoweringSpec, values: Mapping[str, Any]
    ) -> ExecutionResult:
        try:
            bindings = spec.input_binding_map
            inputs = [
                np.asarray(values[bindings[tensor.name]])
                for tensor in spec.program.function.inputs
            ]
        except KeyError as exc:
            return ExecutionResult(
                "scalar",
                False,
                error=f"missing scalar input value {exc.args[0]!r}",
            )
        try:
            broadcast_axes = scalar_broadcast_axes(spec.program)
            execution = execute_on_cpu_outputs(
                spec.program.function,
                spec.dims_map,
                inputs,
                broadcast_axes=broadcast_axes,
            )
            logical_outputs = execution.outputs
            output_bindings = spec.output_binding_map
            outputs = {
                output_bindings[name]: np.asarray(value)
                for name, value in logical_outputs.items()
            }
            if set(outputs) != set(output_bindings.values()):
                raise ValueError("scalar lowering did not produce every bound output")
            return ExecutionResult(
                "scalar",
                True,
                outputs,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "path": "IntentProgram->intent MLIR->linalg/loops->LLVM CPU",
                    "function": spec.program.function.name,
                    "final_artifact_digest": spec.final_artifact_digest,
                    "gate1_j_digest": spec.gate1_j_digest,
                    "intent_j_digest": spec.program.j_digest,
                    "intent_contract_digest": spec.program.contract_digest,
                    "intent_program_digest": spec.program.program_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "broadcast_consumer_count": len(broadcast_axes),
                    "target_backend_in_verdict": False,
                    "execution_evidence": dict(execution.metadata),
                },
            )
        except (CPUBackendUnsupportedError, NotImplementedError) as exc:
            return ExecutionResult(
                "scalar",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "final_artifact_digest": spec.final_artifact_digest,
                    "scalar_spec_digest": spec.spec_digest,
                },
            )
        except CPUABIValidationError as exc:
            return ExecutionResult(
                "scalar",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "final_artifact_digest": spec.final_artifact_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "failure_class": "abi_validation",
                },
            )
        except Exception as exc:
            return ExecutionResult(
                "scalar",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "final_artifact_digest": spec.final_artifact_digest,
                    "scalar_spec_digest": spec.spec_digest,
                },
            )


__all__ = [
    "SCALAR_LOWERING_PROTOCOL",
    "ScalarLoweringRunner",
    "ScalarLoweringSpec",
    "scalar_broadcast_axes",
]
