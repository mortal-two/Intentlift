"""SCRC Gate2 source/replay and source/scalar differential execution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import ResultRelationKind, SemanticArchive

from .compare import compare_outputs, validate_j_inputs
from .model import ExecutionResult, Gate2Result, Gate2Status, Gate2SuiteResult
from .scalar import (
    ScalarLoweringRequest,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
)
from .triton import TritonExecutionSpec, TritonRunner


GATE2_EXECUTION_PROTOCOL = "source_replay_j_relation_v4"


def _input_digest(values: Mapping[str, Any]) -> str:
    """Hash exact trial inputs without lossy JSON float serialization."""

    digest = hashlib.sha256()
    digest.update(b"intentir-gate2-inputs-v1\0")
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise ValueError("Gate2 input names must be non-empty strings")
        value = np.asarray(values[name])
        if value.dtype.hasobject:
            raise ValueError(f"Gate2 input {name!r} has object dtype")
        dense = np.ascontiguousarray(value)
        header = json.dumps(
            {"name": name, "dtype": dense.dtype.str, "shape": list(dense.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = dense.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _clone_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Give every Gate2 leg an isolated view of the committed pre-state."""

    result: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            result[name] = value.copy()
        else:
            result[name] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class Gate2Case:
    case_id: str
    j: SemanticArchive
    values: Mapping[str, Any]
    triton: TritonExecutionSpec
    scalar: ScalarLoweringRequest
    output_keys: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("Gate2 case requires the canonical J object")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")
        if self.scalar.j is not self.j:
            raise ValueError(
                "Gate2 RefEval request and comparator must receive the same J object"
            )
        if isinstance(self.scalar, ScalarLoweringSpec):
            scalar_outputs = set(self.scalar.output_binding_map.values())
            if set(self.output_keys) != scalar_outputs:
                raise ValueError(
                    "Gate2 compared outputs differ from scalar output bindings"
                )
        if len(self.output_keys) != len(set(self.output_keys)):
            raise ValueError("Gate2 output keys must be unique")


@dataclass(frozen=True)
class Gate2Suite:
    suite_id: str
    cases: tuple[Gate2Case, ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite case_id values must be unique")


class Gate2Runner:
    def __init__(
        self,
        *,
        source_runner: TritonRunner | None = None,
        replay_runner: TritonRunner | None = None,
        scalar_runner: ScalarLoweringRunner | None = None,
    ) -> None:
        self.source_runner = source_runner or TritonRunner()
        self.replay_runner = replay_runner or TritonRunner()
        self.scalar_runner = scalar_runner or ScalarLoweringRunner()

    def run(
        self,
        source: KernelSource,
        replay: KernelSource,
        case: Gate2Case,
    ) -> Gate2Result:
        input_digest = _input_digest(case.values)
        if isinstance(case.scalar, ScalarLoweringSpec):
            try:
                validate_j_inputs(
                    case.j,
                    case.scalar.input_binding_map,
                    case.values,
                )
            except (NotImplementedError, ValueError) as exc:
                error = f"J input precondition failed: {type(exc).__name__}: {exc}"
                metadata = {
                    "protocol": GATE2_EXECUTION_PROTOCOL,
                    "failure_class": "j_input_precondition",
                }
                return Gate2Result(
                    case.case_id,
                    Gate2Status.EXECUTION_FAILED,
                    ExecutionResult("source", False, error=error, metadata=metadata),
                    ExecutionResult("replay", False, error=error, metadata=metadata),
                    ExecutionResult("scalar", False, error=error, metadata=metadata),
                    input_digest=input_digest,
                    metadata=dict(case.metadata),
                )

        source_result = self.source_runner.run(
            "source", source, case.triton, _clone_values(case.values)
        )
        replay_result = self.replay_runner.run(
            "replay", replay, case.triton, _clone_values(case.values)
        )
        scalar_result = (
            case.scalar.execution_result()
            if isinstance(case.scalar, ScalarLoweringUnavailable)
            else self.scalar_runner.run(case.scalar, _clone_values(case.values))
        )
        source_replay = (
            compare_outputs(
                "source",
                source_result.outputs,
                "replay",
                replay_result.outputs,
                j=case.j,
                input_bindings=case.scalar.input_binding_map,
                output_bindings=case.scalar.output_binding_map,
                inputs=case.values,
            )
            if source_result.ok
            and replay_result.ok
            and isinstance(case.scalar, ScalarLoweringSpec)
            else None
        )
        source_scalar = (
            compare_outputs(
                "source",
                source_result.outputs,
                "scalar",
                scalar_result.outputs,
                j=case.j,
                input_bindings=case.scalar.input_binding_map,
                output_bindings=case.scalar.output_binding_map,
                inputs=case.values,
            )
            if source_result.ok
            and scalar_result.ok
            and isinstance(case.scalar, ScalarLoweringSpec)
            else None
        )

        if not source_result.ok or not replay_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        elif scalar_result.unsupported:
            status = Gate2Status.SCALAR_UNSUPPORTED
        elif not scalar_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        else:
            assert source_replay is not None and source_scalar is not None
            if source_replay.passed and source_scalar.passed:
                status = Gate2Status.PASS
            elif any(
                item.kind is ResultRelationKind.ONE_OF_UPDATES
                for item in case.j.boundary.result_relations
            ):
                # Pair failure may mean source, replay, scalar, or multiple legs
                # fell outside R_J.  Preserve per-leg details without falsely
                # attributing a set-valued failure to pairwise inequality.
                status = Gate2Status.RESULT_RELATION_DIVERGED
            elif not source_replay.passed and not source_scalar.passed:
                status = Gate2Status.REPLAY_AND_SCALAR_DIVERGED
            elif not source_replay.passed:
                status = Gate2Status.REPLAY_DIVERGED
            else:
                status = Gate2Status.SCALAR_DIVERGED
        return Gate2Result(
            case.case_id,
            status,
            source_result,
            replay_result,
            scalar_result,
            source_replay,
            source_scalar,
            input_digest,
            dict(case.metadata),
        )

    def run_suite(
        self,
        source: KernelSource,
        replay: KernelSource,
        suite: Gate2Suite,
    ) -> Gate2SuiteResult:
        """Run every matrix member; one failure keeps the suite from passing."""

        return Gate2SuiteResult(
            suite.suite_id,
            tuple(self.run(source, replay, case) for case in suite.cases),
        )


__all__ = [
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Runner",
    "Gate2Suite",
]
