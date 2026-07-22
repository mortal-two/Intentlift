"""SCRC Gate2 three-way execution and external-Q pairwise closure."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import SemanticArchive

from .compare import compare_outputs, validate_j_inputs
from .model import ExecutionResult, Gate2Result, Gate2Status, Gate2SuiteResult
from .policy import (
    ComparisonPolicy,
    ComparisonPolicyError,
    DEFAULT_GATE2_POLICY,
)
from .scalar import (
    ScalarLoweringRequest,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
)
from .triton import TritonExecutionSpec, TritonRunner


GATE2_EXECUTION_PROTOCOL = "source_replay_j_three_edges_q_v5"


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
    refeval: ScalarLoweringRequest
    output_keys: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("Gate2 case requires the canonical J object")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")
        if self.refeval.j is not self.j:
            raise ValueError(
                "Gate2 RefEval request and comparator must receive the same J object"
            )
        if isinstance(self.refeval, ScalarLoweringSpec):
            reference_outputs = set(self.refeval.output_binding_map.values())
            if set(self.output_keys) != reference_outputs:
                raise ValueError(
                    "Gate2 compared outputs differ from RefEval output bindings"
                )
        if len(self.output_keys) != len(set(self.output_keys)):
            raise ValueError("Gate2 output keys must be unique")


@dataclass(frozen=True)
class Gate2Suite:
    suite_id: str
    cases: tuple[Gate2Case, ...]
    c_values: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case")
        if [name for name, _ in self.c_values] != sorted(
            name for name, _ in self.c_values
        ) or len({name for name, _ in self.c_values}) != len(self.c_values):
            raise ValueError("Gate2 suite c0 bindings must be unique and sorted")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite case_id values must be unique")


class Gate2Runner:
    def __init__(
        self,
        *,
        source_runner: TritonRunner | None = None,
        replay_runner: TritonRunner | None = None,
        refeval_runner: ScalarLoweringRunner | None = None,
        comparison_policy: ComparisonPolicy | None = None,
    ) -> None:
        self.source_runner = source_runner or TritonRunner()
        self.replay_runner = replay_runner or TritonRunner()
        self.refeval_runner = refeval_runner or ScalarLoweringRunner()
        self.comparison_policy = comparison_policy or DEFAULT_GATE2_POLICY

    def run(
        self,
        source: KernelSource,
        replay: KernelSource,
        case: Gate2Case,
    ) -> Gate2Result:
        input_digest = _input_digest(case.values)
        q_metadata = {
            "protocol": GATE2_EXECUTION_PROTOCOL,
            "comparison_policy": self.comparison_policy.to_json(),
        }
        try:
            self.comparison_policy.validate_j(case.j)
        except ComparisonPolicyError as exc:
            error = f"Gate2 comparison policy rejected J: {exc}"
            failed = ExecutionResult(
                "source", False, error=error, metadata=q_metadata
            )
            return Gate2Result(
                case_id=case.case_id,
                status=Gate2Status.COMPARISON_UNSUPPORTED,
                source=failed,
                replay=ExecutionResult(
                    "replay", False, error=error, metadata=q_metadata
                ),
                reference=ExecutionResult(
                    "j", False, error=error, metadata=q_metadata
                ),
                input_digest=input_digest,
                metadata={**dict(case.metadata), **q_metadata},
            )
        if isinstance(case.refeval, ScalarLoweringSpec):
            try:
                validate_j_inputs(
                    case.j,
                    case.refeval.input_binding_map,
                    case.values,
                )
            except (NotImplementedError, ValueError) as exc:
                error = f"J input precondition failed: {type(exc).__name__}: {exc}"
                metadata = {
                    "protocol": GATE2_EXECUTION_PROTOCOL,
                    "failure_class": "j_input_precondition",
                }
                return Gate2Result(
                    case_id=case.case_id,
                    status=Gate2Status.EXECUTION_FAILED,
                    source=ExecutionResult("source", False, error=error, metadata=metadata),
                    replay=ExecutionResult("replay", False, error=error, metadata=metadata),
                    reference=ExecutionResult("j", False, error=error, metadata=metadata),
                    input_digest=input_digest,
                    metadata={**dict(case.metadata), **q_metadata},
                )

        source_result = self.source_runner.run(
            "source", source, case.triton, _clone_values(case.values)
        )
        replay_result = self.replay_runner.run(
            "replay", replay, case.triton, _clone_values(case.values)
        )
        reference_result = (
            case.refeval.execution_result()
            if isinstance(case.refeval, ScalarLoweringUnavailable)
            else self.refeval_runner.run(
                case.refeval, _clone_values(case.values)
            )
        )
        comparison_kwargs = (
            {
                "j": case.j,
                "input_bindings": case.refeval.input_binding_map,
                "output_bindings": case.refeval.output_binding_map,
                "inputs": case.values,
                "policy": self.comparison_policy,
            }
            if isinstance(case.refeval, ScalarLoweringSpec)
            else None
        )
        source_replay = (
            compare_outputs(
                "source",
                source_result.outputs,
                "replay",
                replay_result.outputs,
                **comparison_kwargs,
            )
            if source_result.ok
            and replay_result.ok
            and comparison_kwargs is not None
            else None
        )
        source_j = (
            compare_outputs(
                "source",
                source_result.outputs,
                "j",
                reference_result.outputs,
                **comparison_kwargs,
            )
            if source_result.ok
            and reference_result.ok
            and comparison_kwargs is not None
            else None
        )
        replay_j = (
            compare_outputs(
                "replay",
                replay_result.outputs,
                "j",
                reference_result.outputs,
                **comparison_kwargs,
            )
            if replay_result.ok
            and reference_result.ok
            and comparison_kwargs is not None
            else None
        )

        if not source_result.ok or not replay_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        elif reference_result.unsupported:
            status = Gate2Status.REFERENCE_UNSUPPORTED
        elif not reference_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        else:
            assert source_replay is not None and source_j is not None
            assert replay_j is not None
            if source_replay.passed and source_j.passed and replay_j.passed:
                status = Gate2Status.PASS
            else:
                status = Gate2Status.PAIRWISE_DIVERGED
        return Gate2Result(
            case_id=case.case_id,
            status=status,
            source=source_result,
            replay=replay_result,
            reference=reference_result,
            source_replay=source_replay,
            source_j=source_j,
            replay_j=replay_j,
            input_digest=input_digest,
            metadata={**dict(case.metadata), **q_metadata},
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
            suite.c_values,
        )


__all__ = [
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Runner",
    "Gate2Suite",
]
