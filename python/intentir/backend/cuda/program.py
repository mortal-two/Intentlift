"""Typed ``IntentProgram`` lowering boundary for the CUDA backend.

This module is intentionally Gate-agnostic.  It consumes only canonical
Layer-A values, typed Layer-B obligations, the logical ABI and replaceable
Layer-C hints carried by :class:`~intentir.ir.program.IntentProgram`.
Unsupported correctness obligations stop the request before MLIR emission or
CUDA compilation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.backend.protocol import (
    BackendConformanceRecord,
    BackendTarget,
    ComparisonEvidence,
    LoweringRequest,
)
from intentir.frontend import emit_module
from intentir.ir.contracts import (
    AccessContract,
    AccessMode,
    AliasRelation,
    ConflictKind,
    DestinationBinding,
    EffectMode,
    IndexRole,
    ObligationLedger,
    ObligationResult,
    ObligationStatus,
    ScopeKind,
    StorageMutability,
    WriteCoverage,
)
from intentir.ir.model import IntentOp
from intentir.ir.program import IntentProgram

from .pipeline import LoweredProgram, lower_to_cuda
from .selection import LoweringTarget


class CudaProgramLoweringError(ValueError):
    """A typed program cannot be consumed losslessly by the CUDA backend."""


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list) and all(isinstance(item, str) for item in op.result):
        return tuple(op.result)
    return ()


def _unsupported_results(program: IntentProgram) -> dict[str, ObligationResult]:
    return {
        item.obligation_id: ObligationResult(
            item.obligation_id,
            ObligationStatus.UNSUPPORTED,
            f"CUDA has no reviewed consumer for {item.capability}",
        )
        for item in program.obligations
    }


def _set_result(
    results: dict[str, ObligationResult],
    obligation_id: str,
    status: ObligationStatus,
    detail: str,
) -> None:
    if obligation_id not in results:
        raise CudaProgramLoweringError(
            f"CUDA analyzer produced an undeclared obligation {obligation_id!r}"
        )
    results[obligation_id] = ObligationResult(obligation_id, status, detail)


def _unique_writer_proof(
    program: IntentProgram,
    destination: DestinationBinding,
    *,
    accesses: Mapping[str, AccessContract],
    ops: Sequence[IntentOp],
) -> bool:
    """Prove a conservative functional unique-writer realization.

    ``unknown`` is never treated as a default no-conflict declaration.  A
    future implementation may prove injective writes from a typed logical
    index map, but the current contract does not expose enough information to
    do so.  Keep this helper deliberately conservative until that proof exists.
    """

    del program, destination, accesses, ops
    return False


def analyze_program_for_cuda(program: IntentProgram) -> ObligationLedger:
    """Return a complete, fail-closed CUDA disposition for every B obligation."""

    if not isinstance(program, IntentProgram):
        raise CudaProgramLoweringError("CUDA analysis requires IntentProgram")
    results = _unsupported_results(program)
    function = program.function
    ops = _walk_ops(function.body)
    flat = len(ops) == len(function.body) and all(op.op != "loop" for op in ops)
    node_by_id = {
        op.node_id: op for op in ops if op.node_id is not None
    }
    values = {item.name for item in [*function.inputs, *function.outputs]}
    for op in ops:
        values.update(_result_names(op))
    tensor_names = {item.name for item in [*function.inputs, *function.outputs]}
    access_by_id = {
        item.contract_id: item for item in program.boundary.accesses
    }

    for item in program.boundary.accesses:
        supported = (
            flat
            and item.mode in {AccessMode.READ, AccessMode.WRITE}
            and item.index_role is IndexRole.REALIZATION_ONLY
            and item.tensor in tensor_names
            and (item.node_ref is None or item.node_ref in node_by_id)
        )
        if supported:
            _set_result(
                results,
                f"{item.contract_id}:access",
                ObligationStatus.CHECKED,
                "functional tensor lowering realizes the declared logical read/write",
            )
            _set_result(
                results,
                f"{item.contract_id}:index_role",
                ObligationStatus.CHECKED,
                "source address arithmetic is realization-only and is replaced by bounded tensor indexing",
            )

    for item in program.boundary.broadcasts:
        op = node_by_id.get(item.node_ref)
        if (
            flat
            and op is not None
            and isinstance(op.result, str)
            and item.operand in op.operands
            and all(isinstance(axis, int) for axis in item.inserted_semantic_axes)
        ):
            _set_result(
                results,
                f"{item.contract_id}:projection",
                ObligationStatus.CONSUMED,
                "translated to the emitter's explicit operand-to-result broadcast projection",
            )

    logical_arguments = {item.name: item for item in program.interface.arguments}
    return_values = {item.output: item.value for item in function.returns}
    destinations_by_output: dict[str, int] = {}
    for item in program.boundary.destinations:
        destinations_by_output[item.output] = destinations_by_output.get(item.output, 0) + 1
        access = access_by_id.get(item.access_ref)
        argument = logical_arguments.get(item.destination)
        binding_ok = (
            flat
            and destinations_by_output[item.output] == 1
            and return_values.get(item.output) == item.value
            and argument is not None
            and argument.role in {"output", "inout"}
            and argument.tensor_ref == item.output
            and access is not None
            and access.mode is AccessMode.WRITE
            and access.tensor == item.output
        )
        if binding_ok:
            _set_result(
                results,
                f"{item.contract_id}:binding",
                ObligationStatus.CONSUMED,
                "logical return is bound to the declared caller destination",
            )
            coverage_ok = item.coverage is WriteCoverage.FULL
            if coverage_ok:
                _set_result(
                    results,
                    f"{item.contract_id}:coverage",
                    ObligationStatus.CHECKED,
                    "bounded tensor result covers every logical output element exactly once",
                )
            conflict_ok = item.conflict is ConflictKind.UNIQUE_WRITER
            conflict_detail = "explicit unique-writer contract"
            if item.conflict is ConflictKind.UNKNOWN:
                conflict_ok = _unique_writer_proof(
                    program, item, accesses=access_by_id, ops=ops
                )
                conflict_detail = "proved injective write map and alias safety"
            if conflict_ok:
                _set_result(
                    results,
                    f"{item.contract_id}:conflict",
                    ObligationStatus.CHECKED,
                    conflict_detail,
                )

    for item in program.boundary.aliases:
        if item.relation is AliasRelation.NO_ALIAS and flat:
            _set_result(
                results,
                f"{item.contract_id}:relation",
                ObligationStatus.CHECKED,
                "functional result allocation preserves the declared no-alias relation",
            )

    access_effect_modes = {
        AccessMode.READ: EffectMode.READ,
        AccessMode.WRITE: EffectMode.WRITE,
    }
    for item in program.boundary.effects:
        access = access_by_id.get(item.access_ref)
        if (
            flat
            and access is not None
            and access_effect_modes.get(access.mode) is item.mode
            and access.tensor == item.target
            and item.node_ref == access.node_ref
        ):
            _set_result(
                results,
                f"{item.contract_id}:effect",
                ObligationStatus.CHECKED,
                "functional tensor use agrees with the declared read/write effect",
            )
    effect_ids = {item.contract_id for item in program.boundary.effects}
    expected_definitions = {
        item.name for item in function.inputs
    } | {
        name for op in ops for name in _result_names(op)
    }
    scope_by_id = {item.contract_id: item for item in program.boundary.scopes}
    for item in program.boundary.scopes:
        if (
            flat
            and item.kind is ScopeKind.FUNCTION
            and item.parent is None
            and not item.captures
            and set(item.definitions) == expected_definitions
            and set(item.effects) == effect_ids
        ):
            _set_result(
                results,
                f"{item.contract_id}:scope",
                ObligationStatus.CHECKED,
                "single functional scope has closed definitions and effects",
            )

    for item in program.boundary.storage:
        scope = scope_by_id.get(item.scope_ref)
        lifetime_ok = (
            flat
            and scope is not None
            and scope.kind is ScopeKind.FUNCTION
            and set(item.members) <= values
            and not item.crosses_iteration
            and item.init_event == f"scope_entry:{item.scope_ref}"
            and item.last_use_event == f"scope_exit:{item.scope_ref}"
        )
        if lifetime_ok:
            _set_result(
                results,
                f"{item.contract_id}:lifetime",
                ObligationStatus.CHECKED,
                "storage lifetime is exactly the emitted functional invocation",
            )
            if item.mutability in {
                StorageMutability.IMMUTABLE,
                StorageMutability.WRITE_ONCE,
            }:
                _set_result(
                    results,
                    f"{item.contract_id}:mutability",
                    ObligationStatus.CHECKED,
                    "SSA tensor realization preserves immutable/write-once storage",
                )

    ordered = tuple(results[item.obligation_id] for item in program.obligations)
    return ObligationLedger(program.contract_digest, program.obligations, ordered)


def _broadcast_axes(program: IntentProgram) -> dict[str, dict[str, tuple[int, ...]]]:
    nodes = {
        op.node_id: op for op in _walk_ops(program.function.body)
        if op.node_id is not None
    }
    projections: dict[str, dict[str, tuple[int, ...]]] = {}
    for item in program.boundary.broadcasts:
        op = nodes.get(item.node_ref)
        if op is None or not isinstance(op.result, str):
            raise CudaProgramLoweringError(
                f"broadcast {item.contract_id!r} has no single-result Layer-A node"
            )
        if not all(isinstance(axis, int) for axis in item.inserted_semantic_axes):
            raise CudaProgramLoweringError(
                f"broadcast {item.contract_id!r} uses symbolic axes unsupported by CUDA emitter"
            )
        axes = tuple(int(axis) for axis in item.inserted_semantic_axes)
        operands = projections.setdefault(op.result, {})
        previous = operands.get(item.operand)
        if previous is not None and previous != axes:
            raise CudaProgramLoweringError(
                f"broadcast {item.contract_id!r} conflicts with another projection"
            )
        operands[item.operand] = axes
    return projections


def _assert_supported(ledger: ObligationLedger) -> None:
    if not ledger.unsupported:
        return
    by_id = {item.obligation_id: item for item in ledger.required}
    results = {item.obligation_id: item for item in ledger.results}
    details = [
        f"{item_id} ({by_id[item_id].capability}): {results[item_id].detail}"
        for item_id in ledger.unsupported
    ]
    raise CudaProgramLoweringError(
        "CUDA cannot consume required Layer-B obligations before codegen: "
        + "; ".join(details)
    )


@dataclass(frozen=True)
class CudaProgramLowering:
    """Compiled CUDA artifact plus its exact typed hand-off evidence."""

    request: LoweringRequest
    obligations: ObligationLedger
    intent_mlir: str
    artifact: LoweredProgram

    def __post_init__(self) -> None:
        program = self.request.effective_program
        if self.obligations.contract_digest != program.contract_digest:
            raise CudaProgramLoweringError("CUDA obligation ledger is bound to another contract")
        _assert_supported(self.obligations)
        mlir_digest = hashlib.sha256(self.intent_mlir.encode("utf-8")).hexdigest()
        if self.artifact.source_mlir_sha256 != mlir_digest:
            raise CudaProgramLoweringError("compiled CUDA artifact differs from emitted Intent MLIR")

    @property
    def mlir_sha256(self) -> str:
        return hashlib.sha256(self.intent_mlir.encode("utf-8")).hexdigest()

    def conformance_record(
        self,
        *,
        test_domain: Mapping[str, Any],
        comparisons: Sequence[ComparisonEvidence] = (),
    ) -> BackendConformanceRecord:
        program = self.request.effective_program
        return BackendConformanceRecord(
            j_digest=program.j_digest,
            contract_digest=program.contract_digest,
            program_digest=program.program_digest,
            target=self.request.target,
            variant=self.request.variant,
            obligations=self.obligations,
            test_domain=dict(test_domain),
            comparisons=tuple(comparisons),
        )

    def to_record(self) -> dict[str, Any]:
        program = self.request.effective_program
        return {
            "j_digest": program.j_digest,
            "contract_digest": program.contract_digest,
            "program_digest": program.program_digest,
            "target": self.request.target.to_dict(),
            "variant": self.request.variant,
            "mlir_sha256": self.mlir_sha256,
            "obligations": self.obligations.to_dict(),
            "cuda": self.artifact.to_record(),
        }


def lower_program_to_cuda(
    request: LoweringRequest,
    concrete_extents: Mapping[str, int],
) -> CudaProgramLowering:
    """Validate every B obligation, emit exact A/B projection, then compile."""

    if not isinstance(request, LoweringRequest):
        raise CudaProgramLoweringError("CUDA lowering requires LoweringRequest")
    if request.target.family != "cuda":
        raise CudaProgramLoweringError(
            f"CUDA backend cannot lower target family {request.target.family!r}"
        )
    if request.variant not in {"generic", "auto"}:
        raise CudaProgramLoweringError(
            "typed CUDA lowering currently accepts only generic or auto variants"
        )
    program = request.effective_program
    ledger = analyze_program_for_cuda(program)
    _assert_supported(ledger)
    mlir = emit_module(
        program.function,
        dict(concrete_extents),
        broadcast_axes=_broadcast_axes(program),
    )
    target = LoweringTarget(
        backend="cuda",
        chip=request.target.architecture,
        features=frozenset(request.target.features),
    )
    artifact = lower_to_cuda(
        mlir,
        target=target,
        allow_specialized=request.variant == "auto",
    )
    return CudaProgramLowering(request, ledger, mlir, artifact)


def cuda_target(
    architecture: str,
    *,
    features: Sequence[str] = (),
) -> BackendTarget:
    """Construct the target-neutral protocol spelling for this backend."""

    return BackendTarget("cuda", architecture, tuple(features))


__all__ = [
    "CudaProgramLowering", "CudaProgramLoweringError", "analyze_program_for_cuda",
    "cuda_target", "lower_program_to_cuda",
]
