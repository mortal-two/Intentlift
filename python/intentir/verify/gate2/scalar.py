"""Simple backend-neutral ``E_J`` that consumes the complete canonical J."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.backend.cpu.abi import CPUABIValidationError, CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import SemanticArchive

from .model import ExecutionResult


SCALAR_LOWERING_PROTOCOL = "canonical_j_scalar_cpu_mlir_v1"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checked_digest(value: str, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _canonical_int_mapping(value: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for key, item in value.items()
    ):
        raise ValueError("scalar dims must map names to non-negative integers")
    return tuple(sorted(value.items()))


def _canonical_str_mapping(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in value.items()
    ):
        raise ValueError("scalar bindings must map non-empty names")
    return tuple(sorted(value.items()))


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def _node_table(j: SemanticArchive) -> dict[str, IntentOp]:
    by_path: dict[str, IntentOp] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = op
            visit(op.body, f"{path}.body")

    visit(j.intent.body)
    result: dict[str, IntentOp] = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _validate_complete_boundary(j: SemanticArchive) -> str:
    """Consume supported B and reject every relation E_J cannot implement."""

    boundary = j.boundary
    if boundary.domains:
        raise NotImplementedError("E_J does not implement explicit domain relations")
    if boundary.accesses:
        kinds = sorted({item.kind.value for item in boundary.accesses})
        raise NotImplementedError(
            "E_J does not implement explicit access relations: " + ", ".join(kinds)
        )
    if boundary.effects:
        kinds = sorted({item.kind.value for item in boundary.effects})
        raise NotImplementedError(
            "E_J does not implement explicit effect relations: " + ", ".join(kinds)
        )
    if boundary.aliases:
        raise NotImplementedError("E_J does not implement alias relations")
    if any(item.mutability.value == "mutable" for item in j.storage):
        raise NotImplementedError("E_J does not implement general mutable storage")

    nodes = _node_table(j)
    reductions_by_ref = {item.node_ref: item for item in boundary.reductions}
    for node_ref, contract in reductions_by_ref.items():
        op = nodes.get(node_ref)
        if op is None or op.op != "reduce":
            raise ValueError("Layer-B reduction does not target a reduce node")
        fn = op.attrs.get("fn")
        neutral = {
            "sum": 0,
            "mean": 0,
            "prod": 1,
            "any": False,
            "all": True,
        }.get(fn)
        declared = (
            contract.identity
            if contract.identity is not None else contract.masked_contribution
        )
        if fn in {"max", "min"}:
            expected = "NEG_INF" if fn == "max" else "POS_INF"
            if declared != expected:
                raise NotImplementedError(
                    f"E_J requires {expected} padding for {fn}"
                )
        elif neutral is not None and declared != neutral:
            raise NotImplementedError(
                f"E_J cannot eliminate non-neutral {fn!r} boundary padding"
            )

    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item.value for item in boundary.output_mappings}
    if mappings and mappings != returns:
        raise NotImplementedError("E_J output destination relation is incomplete")
    return _digest(boundary.to_json())


def scalar_broadcast_axes(
    j: SemanticArchive,
) -> dict[str, dict[str, tuple[int, ...]]]:
    nodes = _node_table(j)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in j.boundary.broadcasts:
        op = nodes.get(projection.node_ref)
        if op is None or not isinstance(op.result, str):
            raise NotImplementedError("scalar broadcast requires one result consumer")
        if projection.operand not in op.operands:
            raise ValueError("scalar broadcast names a non-consumed operand")
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in projection.axes):
            raise NotImplementedError("E_J does not implement symbolic broadcast axes")
        axes = tuple(int(axis) for axis in projection.axes)
        previous = result.setdefault(op.result, {}).get(projection.operand)
        if previous is not None and previous != axes:
            raise ValueError("conflicting scalar broadcast relations")
        result[op.result][projection.operand] = axes
    return result


@dataclass(frozen=True)
class ScalarLoweringSpec:
    """Invocation request holding the same J object used by Gate1."""

    j: SemanticArchive
    final_artifact_digest: str
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("scalar lowering requires canonical J")
        object.__setattr__(
            self,
            "final_artifact_digest",
            _checked_digest(self.final_artifact_digest, "final artifact digest"),
        )
        dims = dict(self.dims)
        inputs = dict(self.input_bindings)
        outputs = dict(self.output_bindings)
        object.__setattr__(self, "dims", _canonical_int_mapping(dims))
        object.__setattr__(self, "input_bindings", _canonical_str_mapping(inputs))
        object.__setattr__(self, "output_bindings", _canonical_str_mapping(outputs))
        expected_inputs = {item.name for item in self.j.intent.inputs}
        expected_outputs = {item.name for item in self.j.intent.outputs}
        if set(inputs) != expected_inputs:
            raise ValueError("scalar input bindings differ from J inputs")
        if set(outputs) != expected_outputs:
            raise ValueError("scalar output bindings differ from J outputs")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("scalar output destinations must be unique")

    @classmethod
    def create(
        cls,
        j: SemanticArchive,
        *,
        final_artifact_digest: str,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
    ) -> "ScalarLoweringSpec":
        return cls(
            j,
            final_artifact_digest,
            tuple(dims.items()),
            tuple(input_bindings.items()),
            tuple(output_bindings.items()),
        )

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

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
                "j_digest": self.j_digest,
                "dims": self.dims_map,
                "input_bindings": self.input_binding_map,
                "output_bindings": self.output_binding_map,
            }
        )


@dataclass(frozen=True)
class ScalarLoweringUnavailable:
    final_artifact_digest: str
    j_digest: str
    error: str
    failure_class: str = "j_lowering_setup"

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_artifact_digest", _checked_digest(self.final_artifact_digest, "final artifact digest"))
        object.__setattr__(self, "j_digest", _checked_digest(self.j_digest, "J digest"))
        if not self.error:
            raise ValueError("unavailable E_J request requires an error")

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "availability": "unavailable",
                "final_artifact_digest": self.final_artifact_digest,
                "j_digest": self.j_digest,
                "failure_class": self.failure_class,
                "error": self.error,
            }
        )

    def execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            "scalar",
            False,
            error=self.error,
            unsupported=True,
            metadata={
                "protocol": SCALAR_LOWERING_PROTOCOL,
                "final_artifact_digest": self.final_artifact_digest,
                "j_digest": self.j_digest,
                "scalar_spec_digest": self.spec_digest,
                "failure_class": self.failure_class,
            },
        )


ScalarLoweringRequest = ScalarLoweringSpec | ScalarLoweringUnavailable


class ScalarLoweringRunner:
    def run(self, spec: ScalarLoweringSpec, values: Mapping[str, Any]) -> ExecutionResult:
        try:
            inputs = [
                np.asarray(values[spec.input_binding_map[tensor.name]])
                for tensor in spec.j.intent.inputs
            ]
        except KeyError as exc:
            return ExecutionResult("scalar", False, error=f"missing E_J input {exc.args[0]!r}")
        try:
            boundary_digest = _validate_complete_boundary(spec.j)
            broadcast_axes = scalar_broadcast_axes(spec.j)
            execution = execute_on_cpu_outputs(
                spec.j.intent,
                spec.dims_map,
                inputs,
                broadcast_axes=broadcast_axes,
            )
            output_bindings = spec.output_binding_map
            outputs = {
                output_bindings[name]: np.asarray(value)
                for name, value in execution.outputs.items()
            }
            if set(outputs) != set(output_bindings.values()):
                raise ValueError("E_J did not produce every observable output")
            return ExecutionResult(
                "scalar",
                True,
                outputs,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "path": "canonical J -> intent MLIR -> scalar CPU",
                    "function": spec.j.intent.name,
                    "final_artifact_digest": spec.final_artifact_digest,
                    "j_digest": spec.j_digest,
                    "boundary_digest": boundary_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "broadcast_consumer_count": len(broadcast_axes),
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
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                },
            )
        except CPUABIValidationError as exc:
            return ExecutionResult(
                "scalar",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"failure_class": "physical_cpu_abi"},
            )
        except Exception as exc:
            return ExecutionResult(
                "scalar", False, error=f"{type(exc).__name__}: {exc}"
            )


__all__ = [
    "SCALAR_LOWERING_PROTOCOL", "ScalarLoweringRequest",
    "ScalarLoweringRunner", "ScalarLoweringSpec", "ScalarLoweringUnavailable",
    "scalar_broadcast_axes",
]
