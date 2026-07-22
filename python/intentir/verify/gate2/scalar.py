"""Simple backend-neutral ``E_J`` that consumes the complete canonical J."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.backend.cpu.abi import CPUABIValidationError, CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import SemanticArchive

from .model import ExecutionResult


SCALAR_LOWERING_PROTOCOL = "canonical_j_scalar_cpu_mlir_v2"


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

    # Typed bounds/update/result relations were already cross-validated against
    # this exact J by SemanticArchive.  Bounds are checked against concrete
    # inputs below; result relations are consumed by Gate2.  A typed update is
    # recognized here, while the CPU capability preflight remains responsible
    # for rejecting scatter until a real Intent->CPU lowering exists.
    nodes = _node_table(j)
    for item in boundary.index_bounds:
        if nodes.get(item.node_ref) is None:
            raise ValueError("E_J index bounds reference an unknown node")
    for item in boundary.updates:
        if nodes.get(item.node_ref) is None:
            raise ValueError("E_J logical update references an unknown node")
    outputs = {item.name for item in j.intent.outputs}
    if any(item.output not in outputs for item in boundary.result_relations):
        raise ValueError("E_J result relation references an unknown output")
    return _digest(boundary.to_json())


def scalar_index_bounds(j: SemanticArchive) -> frozenset[str]:
    """Return the exact emitter projection authorized by canonical J."""

    return frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in j.boundary.index_bounds
    )


def validate_index_bounds_inputs(
    j: SemanticArchive,
    values: Mapping[str, np.ndarray],
) -> None:
    """Check J's indexed-domain preconditions before native scalar execution."""

    for item in j.boundary.index_bounds:
        if item.data not in values or item.indices not in values:
            raise NotImplementedError(
                "E_J runtime bounds checking currently requires data and indices "
                "to be function inputs"
            )
        data = np.asarray(values[item.data])
        indices = np.asarray(values[item.indices])
        if item.axis >= data.ndim:
            raise ValueError("E_J indexed data rank violates canonical J")
        if indices.dtype.kind not in {"i", "u"}:
            raise ValueError("E_J indices must have an integer dtype")
        if bool(np.any(indices < 0)) or bool(np.any(indices >= data.shape[item.axis])):
            raise ValueError(
                f"J index-bounds precondition failed at node {item.node_ref!r}"
            )


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
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("scalar lowering requires canonical J")
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
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
    ) -> "ScalarLoweringSpec":
        return cls(
            j,
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
                "j_digest": self.j_digest,
                "dims": self.dims_map,
                "input_bindings": self.input_binding_map,
                "output_bindings": self.output_binding_map,
            }
        )


@dataclass(frozen=True)
class ScalarLoweringUnavailable:
    j: SemanticArchive
    error: str
    failure_class: str = "j_lowering_setup"

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("unavailable RefEval request requires canonical J")
        if not self.error:
            raise ValueError("unavailable E_J request requires an error")

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "availability": "unavailable",
                "j_digest": self.j_digest,
                "failure_class": self.failure_class,
                "error": self.error,
            }
        )

    def execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            "j",
            False,
            error=self.error,
            unsupported=True,
            metadata={
                "protocol": SCALAR_LOWERING_PROTOCOL,
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
            return ExecutionResult("j", False, error=f"missing E_J input {exc.args[0]!r}")
        try:
            boundary_digest = _validate_complete_boundary(spec.j)
            inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(spec.j.intent.inputs, inputs, strict=True)
            }
            validate_index_bounds_inputs(spec.j, inputs_by_ref)
            broadcast_axes = scalar_broadcast_axes(spec.j)
            execution = execute_on_cpu_outputs(
                spec.j.intent,
                spec.dims_map,
                inputs,
                broadcast_axes=broadcast_axes,
                proven_index_bounds=scalar_index_bounds(spec.j),
            )
            output_bindings = spec.output_binding_map
            outputs = {
                output_bindings[name]: np.asarray(value)
                for name, value in execution.outputs.items()
            }
            if set(outputs) != set(output_bindings.values()):
                raise ValueError("E_J did not produce every observable output")
            return ExecutionResult(
                "j",
                True,
                outputs,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "path": "canonical J -> intent MLIR -> scalar CPU",
                    "function": spec.j.intent.name,
                    "j_digest": spec.j_digest,
                    "boundary_digest": boundary_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "broadcast_consumer_count": len(broadcast_axes),
                    "index_bounds_count": len(spec.j.boundary.index_bounds),
                    "result_relation_count": len(
                        spec.j.boundary.result_relations
                    ),
                    "execution_evidence": dict(execution.metadata),
                },
            )
        except (CPUBackendUnsupportedError, NotImplementedError) as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                },
            )
        except CPUABIValidationError as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"failure_class": "physical_cpu_abi"},
            )
        except Exception as exc:
            return ExecutionResult(
                "j", False, error=f"{type(exc).__name__}: {exc}"
            )


__all__ = [
    "SCALAR_LOWERING_PROTOCOL", "ScalarLoweringRequest",
    "ScalarLoweringRunner", "ScalarLoweringSpec", "ScalarLoweringUnavailable",
    "scalar_broadcast_axes", "scalar_index_bounds",
    "validate_index_bounds_inputs",
]
