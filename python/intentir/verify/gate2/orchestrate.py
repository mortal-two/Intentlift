"""SCRC Gate2 final-J replay/scalar differential execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource

from .compare import compare_outputs
from .model import Gate2Result, Gate2Status, Gate2SuiteResult, Tolerance
from .scalar import ScalarLoweringRunner, ScalarLoweringSpec
from .triton import TritonExecutionSpec, TritonRunner


GATE2_EXECUTION_PROTOCOL = "final_j_replay_scalar_differential_v2"


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


@dataclass(frozen=True)
class Gate2Case:
    case_id: str
    values: Mapping[str, Any]
    triton: TritonExecutionSpec
    scalar: ScalarLoweringSpec
    replay_artifact_digest: str
    output_keys: tuple[str, ...]
    tolerances: Mapping[str, Tolerance] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")
        if (
            not isinstance(self.replay_artifact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.replay_artifact_digest) is None
        ):
            raise ValueError("Gate2 replay artifact digest must be lowercase SHA-256")
        if self.replay_artifact_digest != self.scalar.final_artifact_digest:
            raise ValueError(
                "Gate2 replay is not bound to the scalar spec's final artifact"
            )
        scalar_outputs = set(self.scalar.output_binding_map.values())
        if set(self.output_keys) != scalar_outputs:
            raise ValueError(
                "Gate2 compared outputs differ from scalar output bindings"
            )
        if len(self.output_keys) != len(set(self.output_keys)):
            raise ValueError("Gate2 output keys must be unique")
        unknown_tolerances = sorted(set(self.tolerances) - set(self.output_keys))
        if unknown_tolerances:
            raise ValueError(
                f"Gate2 tolerances name unknown outputs: {unknown_tolerances}"
            )


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
        replay_runner: TritonRunner | None = None,
        scalar_runner: ScalarLoweringRunner | None = None,
    ) -> None:
        self.replay_runner = replay_runner or TritonRunner()
        self.scalar_runner = scalar_runner or ScalarLoweringRunner()

    def run(
        self,
        replay: KernelSource,
        case: Gate2Case,
    ) -> Gate2Result:
        replay_result = self.replay_runner.run(
            "replay", replay, case.triton, case.values
        )
        scalar_result = self.scalar_runner.run(case.scalar, case.values)
        replay_scalar = (
            compare_outputs(
                "replay",
                replay_result.outputs,
                "scalar",
                scalar_result.outputs,
                case.output_keys,
                case.tolerances,
            )
            if replay_result.ok and scalar_result.ok
            else None
        )

        if scalar_result.unsupported:
            status = Gate2Status.SCALAR_UNSUPPORTED
        elif not replay_result.ok or not scalar_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        else:
            assert replay_scalar is not None
            status = (
                Gate2Status.PASS
                if replay_scalar.passed
                else Gate2Status.REPLAY_SCALAR_MISMATCH
            )
        return Gate2Result(
            case.case_id,
            status,
            replay_result,
            scalar_result,
            replay_scalar,
            _input_digest(case.values),
            dict(case.metadata),
        )

    def run_suite(
        self,
        replay: KernelSource,
        suite: Gate2Suite,
    ) -> Gate2SuiteResult:
        """Run every matrix member; one failure keeps the suite from passing."""

        return Gate2SuiteResult(
            suite.suite_id,
            tuple(self.run(replay, case) for case in suite.cases),
        )


__all__ = [
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Runner",
    "Gate2Suite",
]
