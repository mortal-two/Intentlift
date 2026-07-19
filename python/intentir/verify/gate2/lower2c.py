"""IntentIR production-lowering execution leg for Gate2.

The project currently lowers the Intent dialect through linalg/loops to LLVM
and executes it with ``mlir-runner``.  This module exposes that existing path as
the initial ``lower2c`` Gate2 prototype; it deliberately does not evaluate JSON
with the archived NumPy interpreter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from intentir.backend.cpu import run_on_cpu
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import Gate1Artifact, SemanticArchive

from .model import ExecutionResult


LOWER2C_PROTOCOL = (
    "intent_linalg_cpu_symbolic_iota_integer_readback_"
    "boundary_broadcast_compressed_gather_v5"
)


def _operation_at_path(
    archive: SemanticArchive,
    raw_path: str,
) -> dict[str, Any]:
    path = raw_path
    for prefix in ("j.intent.intent_function.", "intent_function.", "j.intent."):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    matches = list(re.finditer(r"(?:^|\.)body\[(\d+)\]", path))
    if not matches or "".join(match.group(0) for match in matches).replace(
        ".", ""
    ) != path.replace(".", ""):
        raise ValueError(f"unsupported J node path {raw_path!r}")
    operations = archive.intent.to_json()["intent_function"].get("body", [])
    operation: dict[str, Any] | None = None
    for depth, match in enumerate(matches):
        index = int(match.group(1))
        if index >= len(operations):
            raise ValueError(f"J node path is out of range: {raw_path!r}")
        operation = operations[index]
        if depth + 1 < len(matches):
            nested = operation.get("body", [])
            if not isinstance(nested, list):
                raise ValueError(f"J node path has no nested body: {raw_path!r}")
            operations = nested
    assert operation is not None
    return operation


def boundary_broadcast_axes(
    archive: SemanticArchive | None,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Resolve existing Boundary.broadcasts to Intent result/operand axes."""

    if archive is None or not archive.boundary.broadcasts:
        return {}
    paths = {
        reference: path
        for node_id, path in archive.node_ids.items()
        for reference in (node_id, path)
    }
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for broadcast in archive.boundary.broadcasts:
        path = paths.get(broadcast.node_ref)
        if path is None:
            raise ValueError(
                f"broadcast target {broadcast.node_ref!r} has no J node path"
            )
        operation = _operation_at_path(archive, path)
        if operation.get("op") not in {"map_binary", "select"}:
            raise NotImplementedError(
                "lower2c boundary broadcast currently supports map_binary/select "
                f"consumers, got {operation.get('op')!r}"
            )
        op_result = operation.get("result")
        if not isinstance(op_result, str):
            raise NotImplementedError(
                "lower2c boundary broadcast requires a single-result consumer"
            )
        operands = operation.get("operands", [])
        if broadcast.operand not in operands:
            raise ValueError(
                f"broadcast operand {broadcast.operand!r} is not consumed by "
                f"node {broadcast.node_ref!r}"
            )
        if any(isinstance(axis, bool) or not isinstance(axis, int)
               for axis in broadcast.axes):
            raise NotImplementedError(
                "lower2c boundary broadcast currently requires integer axes"
            )
        axes = tuple(int(axis) for axis in broadcast.axes)
        operand_axes = result.setdefault(op_result, {})
        if broadcast.operand in operand_axes:
            previous = operand_axes[broadcast.operand]
            detail = "duplicate" if previous == axes else "conflicting"
            raise ValueError(
                f"{detail} broadcast axes for consumer result/operand "
                f"{op_result!r}/{broadcast.operand!r}"
            )
        operand_axes[broadcast.operand] = axes
    return result


@dataclass(frozen=True)
class Lower2CSpec:
    intent: IntentFunction
    dims: Mapping[str, int]
    input_bindings: Mapping[str, str]
    output_key: str
    semantic_archive: SemanticArchive | None = None

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
        if self.semantic_archive is not None:
            archive = self.semantic_archive
            if not isinstance(archive, SemanticArchive):
                archive = SemanticArchive.from_json(archive)
                object.__setattr__(self, "semantic_archive", archive)
            if archive.intent.to_json() != self.intent.to_json():
                raise ValueError(
                    "lower2c semantic_archive.intent differs from the lowered IntentIR"
                )

    @classmethod
    def from_artifact(
        cls,
        artifact: Gate1Artifact,
        *,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_key: str,
    ) -> "Lower2CSpec":
        """Build the production-lowering leg from validated J, never π/source."""

        if not isinstance(artifact, Gate1Artifact):
            artifact = Gate1Artifact.from_json(artifact)
        return cls(
            intent=artifact.j.intent,
            dims=dims,
            input_bindings=input_bindings,
            output_key=output_key,
            semantic_archive=artifact.j,
        )

    def semantic_contract_metadata(self) -> dict[str, Any]:
        if self.semantic_archive is None:
            return {"semantic_contract": "legacy_intent_only"}
        payload = self.semantic_archive.to_json()
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "semantic_contract": "validated_j_scope_storage_v1",
            "semantic_contract_sha256": digest,
            "semantic_scope_count": len(self.semantic_archive.scopes),
            "semantic_storage_count": len(self.semantic_archive.storage),
        }


class Lower2CRunner:
    def run(self, spec: Lower2CSpec, values: Mapping[str, Any]) -> ExecutionResult:
        try:
            inputs = [
                np.asarray(values[spec.input_bindings[tensor.name]])
                for tensor in spec.intent.inputs
            ]
            broadcast_axes = boundary_broadcast_axes(spec.semantic_archive)
            output = run_on_cpu(
                spec.intent,
                dict(spec.dims),
                inputs,
                broadcast_axes=broadcast_axes,
            )
            return ExecutionResult(
                "lowering",
                True,
                {spec.output_key: np.asarray(output)},
                metadata={
                    "path": "IntentIR->intent MLIR->linalg->LLVM->mlir-runner",
                    "protocol": LOWER2C_PROTOCOL,
                    "function": spec.intent.name,
                    "boundary_broadcast_consumer_count": len(broadcast_axes),
                    **spec.semantic_contract_metadata(),
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


# Archived tests and callers used the private spelling before the CUDA Gate
# adapter shared this canonical Layer-B projection.  Keep it as an alias.
_boundary_broadcast_axes = boundary_broadcast_axes


__all__ = [
    "LOWER2C_PROTOCOL",
    "Lower2CRunner",
    "Lower2CSpec",
    "boundary_broadcast_axes",
]
