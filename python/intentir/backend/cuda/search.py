"""Contract-preserving CUDA lowering-state/schedule search over immutable M0.

Candidate legality is decided only by registered C++ whole-graph probes.  The
optional C object influences enumeration order and nothing else.  Runtime
measurements select among candidates that already passed structural admission
and compiled to cubins; measurements are never used as a correctness oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.pipeline import (
    LoweredProgram,
    lower_to_cuda,
    lower_to_cuda_fused,
)
from intentir.backend.cuda.schedules import (
    ELEMENTWISE_FAST_PATH_ID,
    PADDED_ROW_FAST_PATH_ID,
    ROW_FAST_PATH_ID,
    ElementwiseSchedule,
    PaddedRowSchedule,
    RowSchedule,
)
from intentir.backend.contract_capabilities import (
    B_CAP_RANK0_HOST_CAPTURE,
    B_CAP_READONLY_INPUTS,
    derive_search_capabilities,
)
from intentir.backend.materialize import BackendSearchState, CPrior
from intentir.backend.ablation import SearchAxes
from intentir.backend.search_and_close import (
    BackendClosureError,
    BackendClosureReceipt,
    CandidateQDResult,
    ConcreteLoweringState,
    ObservationClass,
    SearchAndCloseController,
)
from intentir.ir.production_snapshot import content_digest

from .program import analyze_j_for_cuda
from .selection import DEFAULT_PASSES, LoweringTarget, select_variant


SEARCH_PROTOCOL = "intentir.backend.cuda.measured-search.v3"
CACHE_PROTOCOL = "intentir.backend.cuda.performance-cache.v2"
CONTRACT_VIEW_VERSION = "intentir.backend.cuda.contract-view.v2"

B_TRANSFORM_EPILOGUE_LOAD_MOTION = (
    "cuda.b.epilogue_readonly_load_motion.v1"
)
B_TRANSFORM_RANK0_HOST_CAPTURE = "cuda.b.rank0_host_value_capture.v1"
B_TRANSFORM_UNIFORM_HOST_CAPTURE = "cuda.b.uniform_host_value_capture.v1"
LINALG_FUSED_FAST_PATH_ID = "cuda.linalg.elementwise-fused.v1"


Benchmark = Callable[[LoweredProgram, "CudaSearchCandidate"], float]


class CudaContractError(ValueError):
    """Raised before candidate compilation when complete Layer B is unsupported."""


def _lowering_path_kind_for_fast_path(fast_path_id: str) -> str:
    """Map registered pass identities to stable, non-semantic path labels."""

    mapping = {
        ELEMENTWISE_FAST_PATH_ID: "elementwise",
        PADDED_ROW_FAST_PATH_ID: "padded_row_reduction",
        ROW_FAST_PATH_ID: "row_reduction",
        "cuda.row_scan.sparsemax.sorted_prefix.v1": "row_scan",
    }
    # Unknown registered paths keep their exact compiler identity.  This avoids
    # inventing a semantic "family" or a kernel-name recipe merely to group
    # structurally different C++ lowering paths.
    return mapping.get(fast_path_id, fast_path_id)


def derive_default_fast_path_id(state: BackendSearchState) -> str | None:
    """Resolve E1's first C++-admitted pass identity from J and target only."""

    derive_cuda_contract_view(state)
    selection = select_variant(
        state.base_ir,
        target=LoweringTarget(
            backend="cuda",
            chip=state.target.architecture,
            features=frozenset(state.target.features),
        ),
    )
    return selection.fast_path_id


def derive_default_lowering_path_kind(state: BackendSearchState) -> str:
    """Resolve E1's admitted path label without compiling a target binary."""

    fast_path_id = derive_default_fast_path_id(state)
    return (
        "generic"
        if fast_path_id is None
        else _lowering_path_kind_for_fast_path(fast_path_id)
    )


@dataclass(frozen=True)
class CudaContractView:
    """Ephemeral B- and target-conditioned admission view of immutable J.

    The facts are derived by the existing CUDA Layer-B consumer.  They are not
    a second semantic object and never own or rewrite any part of J.
    """

    j_digest: str
    target: Mapping[str, Any]
    consumed: tuple[str, ...]
    unsupported: tuple[str, ...]
    capabilities: tuple[str, ...]
    boundary_sha256: str
    boundary_projection: Mapping[str, Any] | None
    version: str = CONTRACT_VIEW_VERSION

    @property
    def supported(self) -> bool:
        return not self.unsupported

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "version": self.version,
                "j_digest": self.j_digest,
                "target": dict(self.target),
                "consumed": list(self.consumed),
                "unsupported": list(self.unsupported),
                "capabilities": list(self.capabilities),
                "boundary_sha256": self.boundary_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "digest": self.digest,
            "j_digest": self.j_digest,
            "target": dict(self.target),
            "supported": self.supported,
            "consumed": list(self.consumed),
            "unsupported": list(self.unsupported),
            "capabilities": list(self.capabilities),
            "boundary_sha256": self.boundary_sha256,
            "boundary_projection": (
                dict(self.boundary_projection)
                if self.boundary_projection is not None
                else None
            ),
        }


def derive_cuda_contract_view(state: BackendSearchState) -> CudaContractView:
    """Consume complete Layer B before enumeration or target compilation."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA ContractView requires BackendSearchState")
    analysis = analyze_j_for_cuda(
        state.base.j,
        materialization=state.base,
    )
    view = CudaContractView(
        j_digest=state.base.j_digest,
        target=state.target.to_dict(),
        consumed=tuple(analysis["consumed"]),
        unsupported=tuple(analysis["unsupported"]),
        capabilities=derive_search_capabilities(state.base),
        boundary_sha256=analysis["boundary_sha256"],
        boundary_projection=analysis["boundary_projection"],
    )
    if not view.supported:
        raise CudaContractError(
            "CUDA search cannot consume complete Layer B: "
            + ", ".join(view.unsupported)
        )
    return view


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolchain_fingerprint() -> str:
    files = {
        "intent_opt": str(backend_compile.INTENT_OPT),
        "mlir_opt": str(backend_compile.MLIR_OPT),
        "ptxas": str(Path(backend_compile.CUDA_BIN) / "ptxas"),
    }
    payload = {
        "chip": backend_compile.CHIP,
        "files": {
            name: {"path": path, "sha256": _sha_file(path)}
            for name, path in files.items()
        },
    }
    return content_digest(payload)


@dataclass(frozen=True)
class CudaSearchCandidate:
    candidate_id: str
    lowering_path_kind: str
    fast_path_id: str | None
    pass_argument: str | None
    schedule: Mapping[str, int | bool | str]
    c_order_score: float
    b_transformations: tuple[str, ...] = ()
    required_b_capabilities: tuple[str, ...] = ()
    requested_trace: tuple[str, ...] = ()
    requested_observation_class: ObservationClass = (
        ObservationClass.HARD_PRESERVING
    )
    compiler_pipeline: str = "direct"

    @property
    def generic(self) -> bool:
        return self.compiler_pipeline == "generic_materialized"

    @property
    def upstream_fused(self) -> bool:
        return self.compiler_pipeline == "linalg_fused"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "lowering_path_kind": self.lowering_path_kind,
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "schedule": dict(self.schedule),
            "c_order_score": self.c_order_score,
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(
                self.required_b_capabilities
            ),
            "requested_trace": list(self.requested_trace),
            "requested_observation_class": (
                self.requested_observation_class.value
            ),
            "compiler_pipeline": self.compiler_pipeline,
        }

    def concrete_state(self, program: LoweredProgram) -> ConcreteLoweringState:
        receipt = program.transformation_receipt
        trace = (
            receipt.transformation_trace
            if receipt is not None
            else self.requested_trace
        )
        if not trace:
            trace = (f"cuda.{self.lowering_path_kind}.v1",)
        topology_digest = (
            receipt.topology_digest
            if receipt is not None
            else content_digest(
                {
                    "lowering_path_kind": self.lowering_path_kind,
                    "transformation_trace": list(trace),
                }
            )
        )
        observation_class = (
            ObservationClass.OBSERVATION_CHANGING
            if receipt is not None and receipt.observation_changing
            else self.requested_observation_class
        )
        mlir_digest = program.concrete_mlir_sha256
        state_id = "cuda.state." + content_digest(
            {
                "mlir_digest": mlir_digest,
                "executable_digest": program.executable_digest,
            }
        )[:20]
        return ConcreteLoweringState(
            state_id=state_id,
            lowering_path_kind=(
                receipt.lowering_path_kind
                if receipt is not None
                else self.lowering_path_kind
            ),
            transformation_trace=tuple(trace),
            schedule=self.schedule,
            observation_class=observation_class,
            mlir_digest=mlir_digest,
            topology_digest=topology_digest,
            executable_digest=program.executable_digest,
            compiler_record=program.to_record(),
        )


@dataclass(frozen=True)
class SearchAttempt:
    candidate: CudaSearchCandidate
    status: str
    reason: str
    compile_ms: float | None = None
    latency_us: float | None = None
    cache_hit: bool = False
    lowering_record: Mapping[str, Any] | None = None
    concrete_state: ConcreteLoweringState | None = None
    qd_search: CandidateQDResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "compile_ms": self.compile_ms,
            "latency_us": self.latency_us,
            "cache_hit": self.cache_hit,
            "lowering": (
                dict(self.lowering_record)
                if self.lowering_record is not None
                else None
            ),
            "concrete_state": (
                self.concrete_state.to_dict()
                if self.concrete_state is not None
                else None
            ),
            "qd_search": (
                self.qd_search.to_dict()
                if self.qd_search is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CudaSearchResult:
    state: BackendSearchState
    contract_view: CudaContractView
    attempts: tuple[SearchAttempt, ...]
    winner: CudaSearchCandidate | None
    winner_program: LoweredProgram | None
    winner_latency_us: float | None
    toolchain_digest: str
    benchmark_key: str | None
    elapsed_ms: float
    time_to_best_ms: float | None
    finalist_count: int
    finalist_remeasurements: int
    b_transformations_enabled: bool
    winner_state: ConcreteLoweringState | None = None
    backend_closure: BackendClosureReceipt | None = None
    backend_closure_error: str | None = None
    ablation_axes: SearchAxes | None = None
    fixed_lowering_path_kind: str | None = None
    fixed_fast_path_id: str | None = None
    hard_admission: tuple[Mapping[str, Any], ...] = ()
    candidate_universe_count: int = 0
    candidate_budget: int = 0
    budget_consumed: int = 0

    def to_record(self) -> dict[str, Any]:
        best_latency: float | None = None
        best_candidate: str | None = None
        curve: list[dict[str, Any]] = []
        consumed_budget = 0
        for attempt_index, attempt in enumerate(self.attempts, start=1):
            if attempt.status != "duplicate_state":
                consumed_budget += 1
            improved = (
                attempt.latency_us is not None
                and (
                    best_latency is None
                    or attempt.latency_us < best_latency
                )
            )
            if improved:
                best_latency = attempt.latency_us
                best_candidate = attempt.candidate.candidate_id
            curve.append(
                {
                    "candidate_attempt": attempt_index,
                    "candidate_budget": consumed_budget,
                    "attempt_budget": consumed_budget,
                    "candidate_id": attempt.candidate.candidate_id,
                    "status": attempt.status,
                    "latency_us": attempt.latency_us,
                    "improved": improved,
                    "best_latency_us": best_latency,
                    "best_candidate_id": best_candidate,
                }
            )

        def first_attempt(predicate: Callable[[SearchAttempt], bool]) -> int | None:
            return next(
                (
                    index
                    for index, attempt in enumerate(self.attempts, start=1)
                    if predicate(attempt)
                ),
                None,
            )

        admitted = tuple(
            item
            for item in self.attempts
            if item.concrete_state is not None
            and item.status in {"compiled", "measured", "benchmark_failed"}
        )
        actual_receipts = tuple(
            receipt
            for item in admitted
            for receipt in (
                (
                    item.concrete_state.compiler_record.get(
                        "transformation_receipt"
                    )
                    if item.concrete_state is not None
                    else None
                ),
            )
            if isinstance(receipt, Mapping)
        )
        reduction_topologies = sorted(
            {
                str(receipt["reduction_topology"])
                for receipt in actual_receipts
                if receipt.get("reduction_topology") not in {None, "none"}
            }
        )
        rows_per_block_values = sorted(
            {
                int(receipt.get("rows_per_block", 1))
                for receipt in actual_receipts
            }
        )
        actual_fma_counts = [
            int(receipt.get("fma_count", 0)) for receipt in actual_receipts
        ]
        approximation_modes = sorted(
            {
                str(receipt.get("approximation_mode"))
                for receipt in actual_receipts
                if receipt.get("approximation_mode") not in {None, "none"}
            }
        )
        status_counts = {
            status: sum(item.status == status for item in self.attempts)
            for status in (
                "compiled",
                "measured",
                "rejected",
                "compile_failed",
                "duplicate_state",
                "qd_rejected",
                "missing_qd",
                "benchmark_failed",
            )
        }
        admitted_pipeline_ids = sorted(
            {
                str(item.concrete_state.compiler_record.get("pipeline_id"))
                for item in admitted
                if item.concrete_state is not None
                and item.concrete_state.compiler_record.get("pipeline_id")
            }
        )
        materialized_kernel_counts = [
            int(item.concrete_state.compiler_record.get("kernel_count", 0))
            for item in admitted
            if item.concrete_state is not None
            and item.concrete_state.compiler_record.get("pipeline_id")
            == "cuda.pipeline.linalg-materialized.v1"
        ]
        whole_graph_kernel_counts = [
            int(item.concrete_state.compiler_record.get("kernel_count", 0))
            for item in admitted
            if item.concrete_state is not None
            and item.concrete_state.compiler_record.get("pipeline_id")
            != "cuda.pipeline.linalg-materialized.v1"
        ]
        fusion_pair_observed = (
            any(count > 1 for count in materialized_kernel_counts)
            and any(count == 1 for count in whole_graph_kernel_counts)
        )

        return {
            "protocol": SEARCH_PROTOCOL,
            "materialization": self.state.base.to_record(),
            "attach_c": self.state.to_record(),
            "contract_view": self.contract_view.to_dict(),
            "toolchain_digest": self.toolchain_digest,
            "benchmark_key": self.benchmark_key,
            "attempts": [item.to_dict() for item in self.attempts],
            "hard_admission": [dict(item) for item in self.hard_admission],
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "winner_latency_us": self.winner_latency_us,
            "winner_state": (
                self.winner_state.to_dict()
                if self.winner_state is not None
                else None
            ),
            "backend_closure": (
                self.backend_closure.to_dict()
                if self.backend_closure is not None
                else None
            ),
            "backend_closure_error": self.backend_closure_error,
            "ablation_axes": (
                self.ablation_axes.to_dict()
                if self.ablation_axes is not None
                else None
            ),
            "fixed_lowering_path_kind": self.fixed_lowering_path_kind,
            "fixed_fast_path_id": self.fixed_fast_path_id,
            "registered_state_axes": {
                "reduction_topology": True,
                "fma_contraction": True,
                "precision_mode": False,
                "approximation_mode": True,
                "precision_reason": (
                    "no typed contraction in the frozen 18-kernel catalog"
                ),
            },
            "elapsed_ms": self.elapsed_ms,
            "time_to_best_ms": self.time_to_best_ms,
            "finalist_tournament": {
                "count": self.finalist_count,
                "remeasurements_per_candidate": self.finalist_remeasurements,
                "aggregation": (
                    "median of fresh round-robin measurements; initial ranking "
                    "sample excluded"
                ),
            },
            "b_search": {
                "transformations_enabled": self.b_transformations_enabled,
                "available_capabilities": list(
                    self.contract_view.capabilities
                ),
                "transformation_candidate_count": sum(
                    bool(item.candidate.b_transformations)
                    for item in self.attempts
                ),
            },
            "actual_state_evidence": {
                "admitted_fast_path_ids": sorted(
                    {
                        item.candidate.fast_path_id
                        for item in admitted
                        if item.candidate.fast_path_id is not None
                    }
                ),
                "admitted_lowering_path_kinds": sorted(
                    {
                        item.concrete_state.lowering_path_kind
                        for item in admitted
                        if item.concrete_state is not None
                    }
                ),
                "admitted_pipeline_ids": admitted_pipeline_ids,
                "materialized_kernel_counts": materialized_kernel_counts,
                "whole_graph_kernel_counts": whole_graph_kernel_counts,
                "concrete_post_pass_mlir_states": len(
                    {
                        item.concrete_state.mlir_digest
                        for item in admitted
                        if item.concrete_state is not None
                        and item.concrete_state.mlir_digest is not None
                    }
                ),
                "concrete_topology_states": len(
                    {
                        item.concrete_state.topology_digest
                        for item in admitted
                        if item.concrete_state is not None
                        and item.concrete_state.topology_digest is not None
                    }
                ),
                "reduction_topologies": reduction_topologies,
                "rows_per_block_values": rows_per_block_values,
                "actual_fma_counts": actual_fma_counts,
                "approximation_modes": approximation_modes,
                "structural_path_expansion_observed": (
                    len(admitted_pipeline_ids) > 1
                    or len(
                        {
                            item.candidate.fast_path_id
                            for item in admitted
                            if item.candidate.fast_path_id is not None
                        }
                    )
                    > 1
                ),
                "reduction_topology_expansion_observed": (
                    len(reduction_topologies) > 1
                ),
                "fma_contraction_observed": any(
                    count > 0 for count in actual_fma_counts
                ),
                "fusion_materialization_pair_observed": (
                    fusion_pair_observed
                ),
                "precision_or_approximation_observed": (
                    len(approximation_modes) > 1
                    or "native_tanh" in approximation_modes
                ),
                "evidence_rule": (
                    "count only compiler-admitted, compiled post-pass MLIR; "
                    "requested toggles and duplicate MLIR do not count"
                ),
            },
            "budget_curve": curve,
            "counts": {
                "candidate_universe": self.candidate_universe_count,
                "candidate_budget": self.candidate_budget,
                "budget_consumed": self.budget_consumed,
                "duplicate_compiles_outside_budget": status_counts[
                    "duplicate_state"
                ],
                "hard_paths_probed": len(self.hard_admission),
                "hard_paths_admitted": sum(
                    item.get("admitted") is True for item in self.hard_admission
                ),
                "hard_paths_rejected": sum(
                    item.get("admitted") is False for item in self.hard_admission
                ),
                "planned": len(self.attempts),
                "admitted": sum(
                    item.status in {"compiled", "measured"}
                    for item in self.attempts
                ),
                "measured": sum(
                    item.latency_us is not None for item in self.attempts
                ),
                "rejected": sum(
                    item.status == "rejected" for item in self.attempts
                ),
                "compile_failed": sum(
                    item.status == "compile_failed" for item in self.attempts
                ),
                "duplicate_state": status_counts["duplicate_state"],
                "qd_rejected": status_counts["qd_rejected"],
                "missing_qd": status_counts["missing_qd"],
                "benchmark_failed": status_counts["benchmark_failed"],
                "status_counts": status_counts,
                "first_admitted_attempt": first_attempt(
                    lambda item: item.status in {"compiled", "measured"}
                ),
                "first_measured_attempt": first_attempt(
                    lambda item: item.latency_us is not None
                ),
                "winner_attempt": (
                    first_attempt(
                        lambda item: self.winner is not None
                        and item.candidate.candidate_id
                        == self.winner.candidate_id
                    )
                    if self.winner is not None
                    else None
                ),
            },
        }


class PerformanceCache:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else (
            Path.home() / ".cache" / "intentir" / "cuda-search-v1.json"
        )
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("protocol") == CACHE_PROTOCOL:
                    self._records = dict(payload.get("records", {}))
            except (OSError, ValueError, TypeError):
                self._records = {}

    @staticmethod
    def key(
        *,
        state: BackendSearchState,
        candidate: CudaSearchCandidate,
        toolchain_digest: str,
        benchmark_key: str,
    ) -> str:
        return content_digest(
            {
                "m0_sha256": state.base.mlir_digest,
                "target": state.target.to_dict(),
                "candidate": {
                    "candidate_id": candidate.candidate_id,
                    "lowering_path_kind": candidate.lowering_path_kind,
                    "fast_path_id": candidate.fast_path_id,
                    "pass_argument": candidate.pass_argument,
                    "schedule": dict(candidate.schedule),
                    "compiler_pipeline": candidate.compiler_pipeline,
                },
                "toolchain_digest": toolchain_digest,
                "benchmark_key": benchmark_key,
            }
        )

    def lookup(self, key: str) -> float | None:
        record = self._records.get(key)
        value = record.get("latency_us") if isinstance(record, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def store(
        self,
        key: str,
        latency_us: float,
        *,
        candidate: CudaSearchCandidate,
    ) -> None:
        if not math.isfinite(latency_us) or latency_us <= 0:
            raise ValueError("performance cache latency must be finite and positive")
        self._records[key] = {
            "latency_us": latency_us,
            "candidate_id": candidate.candidate_id,
            "recorded_unix_ns": time.time_ns(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": CACHE_PROTOCOL,
            "records": self._records,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _candidate_id(
    lowering_path_kind: str,
    fast_path_id: str | None,
    pass_argument: str | None,
) -> str:
    suffix = content_digest(
        {
            "lowering_path_kind": lowering_path_kind,
            "fast_path_id": fast_path_id,
            "pass_argument": pass_argument,
        }
    )[:12]
    return f"{lowering_path_kind}.{suffix}"


def canonical_cuda_candidate(state: BackendSearchState) -> CudaSearchCandidate:
    """Return E1's deterministic J/target-selected compiler state.

    This performs the same registered C++ structural probes as production
    lowering.  It does not enumerate schedules and does not read Source,
    Replay, Q/D, benchmark identity, or measurements.
    """

    derive_cuda_contract_view(state)
    selection = select_variant(
        state.base_ir,
        target=LoweringTarget(
            backend="cuda",
            chip=state.target.architecture,
            features=frozenset(state.target.features),
        ),
    )
    if selection.fast_path_id is None:
        return CudaSearchCandidate(
            _candidate_id("generic", None, None),
            "generic",
            None,
            None,
            {},
            0.0,
            requested_trace=("cuda.generic.intent_to_linalg.v1",),
            compiler_pipeline="generic_materialized",
        )
    assert selection.pass_argument is not None
    path_kind = _lowering_path_kind_for_fast_path(selection.fast_path_id)
    return CudaSearchCandidate(
        _candidate_id(
            path_kind,
            selection.fast_path_id,
            selection.pass_argument,
        ),
        path_kind,
        selection.fast_path_id,
        selection.pass_argument,
        {},
        0.0,
        requested_trace=(selection.fast_path_id,),
        requested_observation_class=ObservationClass.OBSERVATION_CHANGING,
    )


_CUDA_ORDERING_PRIOR_ROLES = frozenset(
    {
        "tile_extent",
        "reduction_tile_extent",
        "execution_warps",
        "constexpr",
    }
)

# CUDA row reductions balance parallelism against per-thread register/dependency
# pressure around a small strip of logical columns.  C describes the logical
# reduction tile, so Retile(C,target) maps it to threads through this strip size
# rather than treating a Source BLOCK_SIZE as either CUDA threads or one exact
# compiler tile.  All neighboring thread counts remain in the same universe and
# are still measured; this constant affects bounded-search order only.
_CUDA_ROW_PREFERRED_ELEMENTS_PER_THREAD = 8


def _numeric_priors(state: BackendSearchState) -> tuple[CPrior, ...]:
    """Return only positive scalar C hints that can rank existing schedules.

    The complete authored record is retained so scoring can respect the role of
    each value.  C remains an ordering hint: this function neither constructs a
    candidate nor participates in hard admission.
    """

    return tuple(
        item
        for item in state.priors
        if item.role in _CUDA_ORDERING_PRIOR_ROLES
        and isinstance(item.value, (int, float))
        and not isinstance(item.value, bool)
        and item.value > 0
    )


def _c_score(
    schedule: Mapping[str, int | bool | str],
    priors: tuple[CPrior, ...],
    *,
    row_elements_per_thread: int = _CUDA_ROW_PREFERRED_ELEMENTS_PER_THREAD,
    elementwise_thread_hint: int | None = None,
) -> float:
    if not priors:
        return 0.0

    def positive_values(names: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(
            float(schedule[name])
            for name in names
            if isinstance(schedule.get(name), (int, float))
            and not isinstance(schedule.get(name), bool)
            and float(schedule[name]) > 0
        )

    distances: list[float] = []
    for prior in priors:
        row_threads = positive_values(("block_threads",))
        is_row_schedule = bool(row_threads and "tile_cols" in schedule)
        is_elementwise_schedule = bool(
            row_threads and "flatten_contiguous" in schedule
        )
        if prior.role == "execution_warps":
            values = positive_values(("num_warps",))
            if not values:
                threads = positive_values(("block_threads",))
                values = tuple(value / 32.0 for value in threads)
        elif prior.role in {
            "reduction_tile_extent",
            "tile_extent",
            "constexpr",
        } and is_row_schedule:
            # A Source reduction tile describes logical row coverage, not a CUDA
            # thread count.  Retile it through the target's preferred logical
            # strip per thread.  For example a 4096-column row maps first to 512
            # threads, while 1024 and 128 columns map to 128 and 32 threads.
            values = tuple(
                thread_count * row_elements_per_thread
                for thread_count in row_threads
            )
        elif prior.role == "reduction_tile_extent":
            values = positive_values(
                ("tile_cols", "tile_rows", "tile_width", "block_threads")
            )
        else:
            # tile_extent and constexpr are both concrete launch/tile hints in
            # accepted C.  They rank only dimensions already present in the
            # candidate; they never create a new dimension or exact state.
            values = positive_values(
                ("tile_cols", "tile_rows", "tile_width", "block_threads")
            )
        if not values:
            continue
        value = float(
            elementwise_thread_hint
            if is_elementwise_schedule
            and elementwise_thread_hint is not None
            and prior.role in {"tile_extent", "constexpr"}
            else prior.value
        )
        distances.append(min(abs(math.log2(candidate / value)) for candidate in values))
    if not distances:
        return float("inf")
    # Every authored C relation contributes.  The former Cartesian minimum
    # discarded all but the easiest-to-match value and produced large zero-score
    # ties, especially when tile and reduction hints coexisted.
    return statistics.fmean(distances)


def _row_elements_per_thread(state: BackendSearchState) -> int:
    """Map J compute density to a CUDA row-retile ordering prior.

    Dense per-element transcendental maps benefit from exposing more lanes to
    hide special-function latency.  Per-row normalization functions such as one
    rsqrt do not trigger this rule.  This is source/name independent and never
    changes hard admission or the candidate universe.
    """

    payload = state.base.j.to_json()["intent"]["intent_function"]["body"]

    def contains_dense_transcendental(value: Any) -> bool:
        if isinstance(value, Mapping):
            if (
                value.get("op") == "map_unary"
                and value.get("attrs", {}).get("fn")
                in {"exp", "log", "sigmoid", "tanh"}
                and len(value.get("type", {}).get("shape", [])) >= 2
            ):
                return True
            return any(contains_dense_transcendental(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_dense_transcendental(item) for item in value)
        return False

    return 4 if contains_dense_transcendental(payload) else 8


def _elementwise_thread_hint(state: BackendSearchState) -> int | None:
    """Retile a logical row packet to enough CUDA threads for target occupancy."""

    extents = state.base.extents
    for output in state.base.j.intent.outputs:
        if len(output.shape) != 2:
            continue
        rows, columns = (
            extents[item] if isinstance(item, str) else item
            for item in output.shape
        )
        if not isinstance(rows, int) or not isinstance(columns, int):
            return None
        # Keep roughly 64K logical threads in flight across independent rows,
        # while never assigning more lanes than the row can use.  This maps
        # 128x1024 to 512 threads and 2048x4096 to 64 threads on the same target.
        desired = math.ceil(65536 / max(1, rows))
        desired = max(64, min(1024, desired))
        desired = 1 << max(6, round(math.log2(desired)))
        row_cap = 1 << max(5, math.ceil(math.log2(max(1, columns))))
        return min(desired, row_cap, 1024)
    return None


def enumerate_cuda_candidates(
    state: BackendSearchState,
    *,
    enable_b_transformations: bool = True,
    axes: SearchAxes | None = None,
    fixed_lowering_path_kind: str | None = None,
    fixed_fast_path_id: str | None = None,
) -> tuple[CudaSearchCandidate, ...]:
    """Enumerate concrete pass states only after complete B admission."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA candidate enumeration requires BackendSearchState")
    if axes is not None:
        enable_b_transformations = axes.enable_local_b_transformations
    if (
        axes is not None
        and not axes.search_structural_states
        and fixed_lowering_path_kind is None
        and fixed_fast_path_id is None
    ):
        raise ValueError(
            "fixed-structure schedule search requires a concrete path identity"
        )
    # A fixed concrete path is the complete E2 search space.  The return logic
    # below excludes the generic materialized pipeline, which would silently
    # turn schedule-only search back into a multi-path experiment.
    contract_view = derive_cuda_contract_view(state)
    capabilities = set(contract_view.capabilities)
    runtime_names = set(state.base.runtime_input_names)
    has_runtime_rank0 = any(
        item.name in runtime_names and not item.shape
        for item in state.base.j.intent.inputs
    )
    has_column_input = any(
        item.name in runtime_names and len(item.shape) == 1
        for item in state.base.j.intent.inputs
    )
    has_uniform_input = any(
        item.name in runtime_names
        and (not item.shape or item.shape == [1])
        for item in state.base.j.intent.inputs
    )
    priors = (
        _numeric_priors(state)
        if axes is None or axes.use_c_prior
        else ()
    )
    row_elements_per_thread = _row_elements_per_thread(state)
    elementwise_thread_hint = _elementwise_thread_hint(state)
    specialized: list[CudaSearchCandidate] = []

    elementwise_schedule_bases = [
        *(ElementwiseSchedule(threads, False, 1)
          for threads in (32, 64, 128, 256, 512, 1024)),
        *(ElementwiseSchedule(threads, True, width)
          for threads in (64, 128, 256)
          for width in (2, 4)),
    ]
    approximation_modes = (
        ("stable-ex2", "native-ptx")
        if axes is None or axes.search_approximation
        else ("stable-ex2",)
    )
    elementwise_schedules = [
        replace(
            base,
            promote_uniform_scalars=promote,
            tanh_approximation=approximation,
        )
        for base in elementwise_schedule_bases
        for promote in ((False, True) if has_uniform_input else (False,))
        for approximation in approximation_modes
    ]
    for schedule in elementwise_schedules:
        values = schedule.to_dict()
        transformations: list[str] = []
        requirements: list[str] = []
        if schedule.promote_uniform_scalars and has_uniform_input:
            transformations.append(B_TRANSFORM_UNIFORM_HOST_CAPTURE)
            requirements.append(B_CAP_READONLY_INPUTS)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "elementwise",
                    ELEMENTWISE_FAST_PATH_ID,
                    schedule.pass_argument,
                ),
                "elementwise",
                ELEMENTWISE_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(
                    values,
                    priors,
                    elementwise_thread_hint=elementwise_thread_hint,
                ),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.elementwise.row.vectorized.v1",
                    *(
                        ("cuda.numeric.tanh_native_ptx.f32.request.v1",)
                        if schedule.tanh_approximation == "native-ptx"
                        else ()
                    ),
                    *transformations,
                ),
                (
                    ObservationClass.OBSERVATION_CHANGING
                    if schedule.tanh_approximation == "native-ptx"
                    else ObservationClass.HARD_PRESERVING
                ),
            )
        )

    row_schedule_bases = [
        *(RowSchedule(threads, 32, vector_width=0, fuse_fma=False)
          for threads in (32, 64, 128, 256, 512, 1024)),
        *(RowSchedule(threads, 64, vector_width=0, fuse_fma=False)
          for threads in (64, 128, 256)),
        *(RowSchedule(threads, 32, vector_width=1, fuse_fma=False)
          for threads in (64, 128, 256)),
        *(RowSchedule(threads, 32, vector_width=4, fuse_fma=False)
          for threads in (32, 64, 128)),
    ]
    if axes is None or axes.search_reduction_topology:
        row_schedule_bases.extend(
            RowSchedule(32, 32, rows_per_block=rows)
            for rows in (2, 4, 8)
        )
    if axes is not None and not axes.search_reduction_topology:
        # E2/N0 freezes the concrete collective topology to the canonical
        # multi-warp hierarchy while still tuning its schedule parameters.
        row_schedule_bases = [
            item for item in row_schedule_bases if item.block_threads > 32
        ]
    fma_modes = (
        (False, True)
        if axes is None or axes.search_fma
        else (False,)
    )
    approximate_div_modes = (
        (False, True)
        if axes is None or axes.search_precision
        else (False,)
    )
    row_schedules = [
        replace(
            schedule,
            fuse_fma=fuse_fma,
            approximate_div=approximate_div,
            promote_rank0_scalars=promote,
            defer_epilogue_col_loads=defer,
        )
        for schedule in row_schedule_bases
        for fuse_fma in fma_modes
        for approximate_div in approximate_div_modes
        for promote in (True, False)
        for defer in (True, False)
    ]
    for schedule in row_schedules:
        values = schedule.to_dict()
        transformations: list[str] = []
        requirements: list[str] = []
        if schedule.promote_rank0_scalars and has_runtime_rank0:
            transformations.append(B_TRANSFORM_RANK0_HOST_CAPTURE)
            requirements.append(B_CAP_RANK0_HOST_CAPTURE)
        if schedule.defer_epilogue_col_loads and has_column_input:
            transformations.append(B_TRANSFORM_EPILOGUE_LOAD_MOTION)
            requirements.append(B_CAP_READONLY_INPUTS)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "row_reduction", ROW_FAST_PATH_ID, schedule.pass_argument
                ),
                "row_reduction",
                ROW_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(
                    values,
                    priors,
                    row_elements_per_thread=row_elements_per_thread,
                ),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.row_reduction.read_once.v1",
                    (
                        "cuda.reduction.subgroup_rows.v1"
                        if schedule.rows_per_block > 1
                        else "cuda.reduction.subgroup_shuffle.v1"
                        if schedule.block_threads == 32
                        else "cuda.reduction.workgroup_hierarchical.v1"
                    ),
                    *(
                        ("cuda.numeric.fma.f32.request.v1",)
                        if schedule.fuse_fma
                        else ()
                    ),
                    *(
                        ("cuda.numeric.div_reciprocal_approx.f32.request.v1",)
                        if schedule.approximate_div
                        else ()
                    ),
                    *transformations,
                ),
                ObservationClass.OBSERVATION_CHANGING,
            )
        )

    padded_schedule_bases = [
        PaddedRowSchedule(32, False),
        PaddedRowSchedule(64, False),
        PaddedRowSchedule(128, False),
        PaddedRowSchedule(32, True),
        PaddedRowSchedule(64, True),
        PaddedRowSchedule(128, True),
    ]
    # The subgroup-feature mapping is a real C++ lowering state: one subgroup
    # owns a padded output row and lanes load contiguous feature elements.  It
    # is only admitted in topology-search arms; E2/N0 remains the fixed
    # serial-row state.  The C++ probe rejects non-power-of-two feature widths.
    if axes is None or axes.search_reduction_topology:
        padded_schedule_bases.extend(
            PaddedRowSchedule(64, False, True, "subgroup-feature")
            for _ in (0,)
        )
        padded_schedule_bases.extend(
            PaddedRowSchedule(128, False, True, "subgroup-feature")
            for _ in (0,)
        )
    for schedule in (
        replace(base, promote_rank0_scalars=promote)
        for base in padded_schedule_bases
        for promote in (True, False)
    ):
        values = schedule.to_dict()
        transformations = []
        requirements = []
        if schedule.promote_rank0_scalars and has_runtime_rank0:
            transformations.append(B_TRANSFORM_RANK0_HOST_CAPTURE)
            requirements.append(B_CAP_RANK0_HOST_CAPTURE)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "padded_row_reduction",
                    PADDED_ROW_FAST_PATH_ID,
                    schedule.pass_argument,
                ),
                "padded_row_reduction",
                PADDED_ROW_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(values, priors),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.padded_row_reduction.masked_short_dot.v1",
                    (
                        "cuda.reduction.padded_subgroup_feature.v1"
                        if schedule.reduction_topology == "subgroup-feature"
                        else "cuda.reduction.padded_serial_row.v1"
                    ),
                    *transformations,
                ),
                ObservationClass.OBSERVATION_CHANGING,
            )
        )

    output_rank = max(
        (len(item.shape) for item in state.base.j.intent.outputs),
        default=1,
    )
    for width in (64, 128, 256):
        tile = (width,) if output_rank <= 1 else (1, width)
        tile_text = "x".join(str(item) for item in tile)
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "linalg_fused",
                    LINALG_FUSED_FAST_PATH_ID,
                    f"linalg-fuse-elementwise-ops;tile={tile_text}",
                ),
                "linalg_fused",
                LINALG_FUSED_FAST_PATH_ID,
                f"linalg-fuse-elementwise-ops;tile={tile_text}",
                {
                    "tile_shape": tile_text,
                    "tile_width": width,
                },
                _c_score({"tile_cols": width}, priors),
                requested_trace=(
                    "cuda.linalg.elementwise_fusion.v1",
                    "cuda.linalg.parallel_loop_tiling.v1",
                ),
                compiler_pipeline="linalg_fused",
            )
        )

    # Every target-compatible C++ whole-graph pass is a real structural search
    # path.  The three high-volume paths above own schedule grids; all other
    # registered passes enter with their compiler default exactly once.  The
    # structural probe still decides admission for the current J, so this is
    # neither a kernel allowlist nor a Python-emitted implementation.
    scheduled_ids = {
        ELEMENTWISE_FAST_PATH_ID,
        PADDED_ROW_FAST_PATH_ID,
        ROW_FAST_PATH_ID,
    }
    for descriptor in DEFAULT_PASSES:
        if descriptor.fast_path_id in scheduled_ids:
            continue
        if state.target.architecture not in descriptor.supported_chips:
            continue
        path_kind = _lowering_path_kind_for_fast_path(
            descriptor.fast_path_id
        )
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    path_kind,
                    descriptor.fast_path_id,
                    descriptor.pass_argument,
                ),
                path_kind,
                descriptor.fast_path_id,
                descriptor.pass_argument,
                {},
                0.0,
                requested_trace=(descriptor.fast_path_id,),
                requested_observation_class=(
                    ObservationClass.OBSERVATION_CHANGING
                ),
            )
        )

    if fixed_fast_path_id is not None:
        specialized = [
            item
            for item in specialized
            if item.fast_path_id == fixed_fast_path_id
        ]
    elif fixed_lowering_path_kind is not None:
        specialized = [
            item
            for item in specialized
            if item.lowering_path_kind == fixed_lowering_path_kind
        ]
    if fixed_fast_path_id is not None and not specialized:
        raise CudaContractError(
            f"fixed CUDA path {fixed_fast_path_id!r} is not registered for the target"
        )
    if (
        fixed_lowering_path_kind not in {None, "generic"}
        and not specialized
    ):
        raise CudaContractError(
            "fixed CUDA lowering path kind "
            f"{fixed_lowering_path_kind!r} has no registered concrete state"
        )

    # A C tile/thread prior has no authority to rank one lowering path above
    # another. Keep path discovery fair under finite budget and use C only to
    # order schedules inside each registered path kind.
    def search_lane(item: CudaSearchCandidate) -> str:
        local = "local" if item.b_transformations else "base"
        if item.lowering_path_kind == "row_reduction":
            topology = (
                "subgroup_rows"
                if int(item.schedule.get("rows_per_block", 1)) > 1
                else "subgroup"
                if int(item.schedule.get("block_threads", 64)) == 32
                else "workgroup"
            )
            if item.schedule.get("approximate_div") is True:
                numeric = "approximate_div"
            elif item.schedule.get("fuse_fma") is True:
                numeric = "fma"
            else:
                numeric = "plain"
            return f"row_reduction.{topology}.{numeric}.{local}"
        if item.lowering_path_kind == "elementwise":
            approximation = str(
                item.schedule.get("tanh_approximation", "stable-ex2")
            )
            return f"elementwise.{approximation}.{local}"
        if item.lowering_path_kind == "padded_row_reduction":
            topology = str(item.schedule.get("reduction_topology", "serial-row"))
            return f"{item.lowering_path_kind}.{topology}.{local}"
        return item.lowering_path_kind

    known_lane_order = (
        "elementwise.stable-ex2.base",
        "row_reduction.workgroup.plain.base",
        "padded_row_reduction.serial-row.base",
        "padded_row_reduction.subgroup-feature.base",
        "elementwise.native-ptx.base",
        "row_reduction.subgroup_rows.plain.base",
        "row_reduction.subgroup.plain.base",
        "row_reduction.workgroup.fma.base",
        "row_reduction.subgroup.fma.base",
        "row_reduction.workgroup.approximate_div.base",
        "row_reduction.subgroup_rows.approximate_div.base",
        "row_reduction.subgroup.approximate_div.base",
        "elementwise.stable-ex2.local",
        "elementwise.native-ptx.local",
        "row_reduction.workgroup.plain.local",
        "row_reduction.subgroup_rows.plain.local",
        "row_reduction.subgroup.plain.local",
        "row_reduction.workgroup.fma.local",
        "row_reduction.subgroup.fma.local",
        "row_reduction.workgroup.approximate_div.local",
        "row_reduction.subgroup_rows.approximate_div.local",
        "row_reduction.subgroup.approximate_div.local",
        "padded_row_reduction.serial-row.local",
        "padded_row_reduction.subgroup-feature.local",
    )
    lanes = {search_lane(item) for item in specialized}
    path_order = (
        *(lane for lane in known_lane_order if lane in lanes),
        *sorted(lanes - set(known_lane_order)),
    )
    by_path = {
        lane: [item for item in specialized if search_lane(item) == lane]
        for lane in path_order
    }
    for candidates in by_path.values():
        candidates.sort(key=lambda item: item.c_order_score)
    specialized = []
    rank = 0
    while any(rank < len(by_path[path]) for path in path_order):
        for path in path_order:
            candidates = by_path[path]
            if rank < len(candidates):
                specialized.append(candidates[rank])
        rank += 1
    generic = CudaSearchCandidate(
        _candidate_id("generic", None, None),
        "generic",
        None,
        None,
        {},
        0.0,
        requested_trace=("cuda.generic.intent_to_linalg.v1",),
        compiler_pipeline="generic_materialized",
    )
    if fixed_fast_path_id is not None:
        return tuple(specialized)
    if fixed_lowering_path_kind is not None:
        if fixed_lowering_path_kind == "generic":
            return (generic,)
        return tuple(specialized)
    return (generic, *specialized)


def admit_cuda_candidate_paths(
    state: BackendSearchState,
    candidates: tuple[CudaSearchCandidate, ...],
) -> tuple[tuple[CudaSearchCandidate, ...], tuple[Mapping[str, Any], ...]]:
    """Run source-blind structural probes before spending tuning budget."""

    grouped: dict[str, list[CudaSearchCandidate]] = {}
    generic_candidates: list[CudaSearchCandidate] = []
    for candidate in candidates:
        if candidate.generic:
            generic_candidates.append(candidate)
            continue
        assert candidate.fast_path_id is not None
        grouped.setdefault(candidate.fast_path_id, []).append(candidate)

    target = LoweringTarget(
        backend="cuda",
        chip=state.target.architecture,
        features=frozenset(state.target.features),
    )
    admitted_ids: set[str] = set()
    records: list[Mapping[str, Any]] = []
    if generic_candidates:
        records.append(
            {
                "path_id": "cuda.pipeline.linalg-materialized.v1",
                "candidate_count": len(generic_candidates),
                "status": "admitted_by_registered_generic_pipeline",
                "admitted": True,
                "probe": None,
            }
        )
    for fast_path_id, path_candidates in grouped.items():
        if fast_path_id == LINALG_FUSED_FAST_PATH_ID:
            admitted_ids.add(fast_path_id)
            records.append(
                {
                    "path_id": fast_path_id,
                    "candidate_count": len(path_candidates),
                    "status": "deferred_to_exact_upstream_mlir_pipeline",
                    "admitted": True,
                    "probe": None,
                }
            )
            continue
        try:
            selection = select_variant(
                state.base_ir,
                target=target,
                only_fast_path=fast_path_id,
            )
        except Exception as error:
            records.append(
                {
                    "path_id": fast_path_id,
                    "candidate_count": len(path_candidates),
                    "status": "probe_failed",
                    "admitted": False,
                    "reason": f"{type(error).__name__}: {error}",
                    "probe": None,
                }
            )
            continue
        admitted = (
            selection.pipeline_kind == "specialized"
            and selection.fast_path_id == fast_path_id
        )
        if admitted:
            admitted_ids.add(fast_path_id)
        records.append(
            {
                "path_id": fast_path_id,
                "candidate_count": len(path_candidates),
                "status": "admitted" if admitted else "not_matched",
                "admitted": admitted,
                "probe": selection.to_dict(),
            }
        )

    admitted_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.generic or candidate.fast_path_id in admitted_ids
    )
    return admitted_candidates, tuple(records)


def compile_cuda_candidate(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
) -> LoweredProgram:
    """Compile one path-admitted request through its registered MLIR pipeline.

    Search, portable export, and any future offline compiler must share this
    dispatcher.  Keeping the generic, upstream-fused, and direct C++ paths in
    one place prevents a transport tool from silently treating a requested
    path as an unknown direct pass or from counting a different compiler state.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA candidate compilation requires BackendSearchState")
    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA candidate compilation requires CudaSearchCandidate")
    if candidate.generic:
        return lower_to_cuda(
            state.base_ir,
            allow_specialized=False,
        )
    if candidate.upstream_fused:
        tile = tuple(
            int(item)
            for item in str(candidate.schedule["tile_shape"]).split("x")
        )
        return lower_to_cuda_fused(
            state.base_ir,
            tile=tile,
            require_single_kernel=True,
        )
    assert candidate.fast_path_id is not None
    assert candidate.pass_argument is not None
    return lower_to_cuda(
        state.base_ir,
        only_fast_path=candidate.fast_path_id,
        pass_overrides={candidate.fast_path_id: candidate.pass_argument},
        require_specialized=True,
    )


def search_cuda(
    state: BackendSearchState,
    *,
    benchmark: Benchmark | None = None,
    benchmark_key: str | None = None,
    budget: int = 96,
    include_generic: bool = True,
    cache: PerformanceCache | None = None,
    finalist_count: int = 16,
    finalist_remeasurements: int = 2,
    enable_b_transformations: bool = True,
    axes: SearchAxes | None = None,
    fixed_lowering_path_kind: str | None = None,
    fixed_fast_path_id: str | None = None,
    qd_controller: SearchAndCloseController | None = None,
    require_backend_closure: bool = False,
) -> CudaSearchResult:
    """Compile and optionally measure a bounded, already-legal candidate set."""

    if budget <= 0:
        raise ValueError("CUDA search budget must be positive")
    if (benchmark is None) != (benchmark_key is None):
        raise ValueError("benchmark and benchmark_key must be provided together")
    if finalist_count <= 0:
        raise ValueError("CUDA search finalist_count must be positive")
    if finalist_remeasurements < 0:
        raise ValueError("CUDA search finalist_remeasurements cannot be negative")
    if axes is not None and budget != axes.candidate_budget:
        raise ValueError("CUDA search budget differs from frozen ablation arm")
    if axes is not None:
        enable_b_transformations = axes.enable_local_b_transformations
    if require_backend_closure and qd_controller is None:
        raise ValueError("formal backend closure requires SearchAndClose")
    started = time.perf_counter_ns()
    toolchain = toolchain_fingerprint()
    performance_cache = cache or PerformanceCache()
    contract_view = derive_cuda_contract_view(state)
    if (
        axes is not None
        and not axes.search_structural_states
        and fixed_lowering_path_kind is None
        and fixed_fast_path_id is None
    ):
        default_fast_path_id = derive_default_fast_path_id(state)
        default_lowering_path_kind = derive_default_lowering_path_kind(state)
        default_probe = select_variant(
            state.base_ir,
            target=LoweringTarget(
                backend="cuda",
                chip=state.target.architecture,
                features=frozenset(state.target.features),
            ),
            only_fast_path=default_fast_path_id,
        )
        if (
            default_probe.pipeline_kind == "specialized"
            and default_probe.fast_path_id == default_fast_path_id
        ):
            fixed_fast_path_id = default_fast_path_id
            fixed_lowering_path_kind = default_lowering_path_kind
            include_generic = False
        else:
            # A requested graph class is not a concrete path identity until its
            # registered C++ matcher admits the current IR.  Keep E2 fixed on
            # the canonical generic state instead of manufacturing a missing
            # specialized trace or leaving the arm without an executable.
            fixed_fast_path_id = None
            fixed_lowering_path_kind = "generic"
            include_generic = True
    planned = enumerate_cuda_candidates(
        state,
        enable_b_transformations=enable_b_transformations,
        axes=axes,
        fixed_lowering_path_kind=fixed_lowering_path_kind,
        fixed_fast_path_id=fixed_fast_path_id,
    )
    candidate_universe = tuple(
        candidate
        for candidate in planned
        if include_generic or not candidate.generic
    )
    admitted_candidates, hard_admission = admit_cuda_candidate_paths(
        state,
        tuple(candidate_universe),
    )
    # Budget counts distinct executable compiler states (plus failed exact
    # candidates), not repeated schedule spellings that C++ lowers to identical
    # post-pass MLIR.  Duplicate detection necessarily happens after the cheap
    # structural/compiler probe; such duplicates remain in the receipt but no
    # longer starve later topology paths under a finite tuning budget.
    candidates = list(admitted_candidates)

    attempts: list[SearchAttempt] = []
    winner: CudaSearchCandidate | None = None
    winner_program: LoweredProgram | None = None
    winner_latency: float | None = None
    time_to_best: float | None = None
    measured_programs: dict[str, LoweredProgram] = {}
    cache_keys: dict[str, str] = {}
    concrete_states: dict[str, ConcreteLoweringState] = {}
    seen_mlir_states: dict[str, str] = {}
    budget_consumed = 0

    for candidate in candidates:
        if budget_consumed >= budget:
            break
        compile_started = time.perf_counter_ns()
        try:
            program = compile_cuda_candidate(state, candidate)
        except ValueError as error:
            budget_consumed += 1
            attempts.append(
                SearchAttempt(
                    candidate,
                    "rejected",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        except Exception as error:  # compiler diagnostics belong in the receipt
            budget_consumed += 1
            attempts.append(
                SearchAttempt(
                    candidate,
                    "compile_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        concrete_state = candidate.concrete_state(program)
        duplicate_of = seen_mlir_states.get(concrete_state.mlir_digest or "")
        if duplicate_of is not None:
            attempts.append(
                SearchAttempt(
                    candidate,
                    "duplicate_state",
                    "concrete post-pass MLIR duplicates " + duplicate_of,
                    compile_ms=compile_ms,
                    lowering_record=program.to_record(),
                    concrete_state=concrete_state,
                )
            )
            continue
        if concrete_state.mlir_digest is not None:
            seen_mlir_states[concrete_state.mlir_digest] = candidate.candidate_id
        budget_consumed += 1
        concrete_states[candidate.candidate_id] = concrete_state

        if (
            benchmark is not None
            and concrete_state.observation_class
            is ObservationClass.OBSERVATION_CHANGING
            and qd_controller is None
        ):
            attempts.append(
                SearchAttempt(
                    candidate,
                    "missing_qd",
                    "observation-changing state requires target-to-RefJ Q/D "
                    "before performance measurement",
                    compile_ms=compile_ms,
                    lowering_record=program.to_record(),
                    concrete_state=concrete_state,
                )
            )
            continue

        qd_search: CandidateQDResult | None = None
        if qd_controller is not None:
            qd_search = qd_controller.filter_candidate(program, concrete_state)
            if not qd_search.passed:
                attempts.append(
                    SearchAttempt(
                        candidate,
                        "qd_rejected",
                        "direct target-to-RefJ comparison failed on D_search",
                        compile_ms=compile_ms,
                        lowering_record=program.to_record(),
                        concrete_state=concrete_state,
                        qd_search=qd_search,
                    )
                )
                continue

        latency: float | None = None
        cache_hit = False
        status = "compiled"
        reason = (
            "complete Layer-B audit, C++ whole-graph admission, and cubin "
            "compilation passed"
        )
        if benchmark is not None:
            assert benchmark_key is not None
            key = performance_cache.key(
                state=state,
                candidate=candidate,
                toolchain_digest=toolchain,
                benchmark_key=benchmark_key,
            )
            cache_keys[candidate.candidate_id] = key
            latency = performance_cache.lookup(key)
            cache_hit = latency is not None
            if latency is None:
                try:
                    latency = float(benchmark(program, candidate))
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError("benchmark returned a non-positive latency")
                except Exception as error:
                    attempts.append(
                        SearchAttempt(
                            candidate,
                            "benchmark_failed",
                            f"{type(error).__name__}: {error}",
                            compile_ms=compile_ms,
                            lowering_record=program.to_record(),
                            concrete_state=concrete_state,
                            qd_search=qd_search,
                        )
                    )
                    continue
                performance_cache.store(key, latency, candidate=candidate)
            status = "measured"
            reason = "performance-only selection over admitted candidate"
            measured_programs[candidate.candidate_id] = program

        attempts.append(
            SearchAttempt(
                candidate,
                status,
                reason,
                compile_ms=compile_ms,
                latency_us=latency,
                cache_hit=cache_hit,
                lowering_record=program.to_record(),
                concrete_state=concrete_state,
                qd_search=qd_search,
            )
        )
        if latency is not None and (
            winner_latency is None or latency < winner_latency
        ):
            winner = candidate
            winner_program = program
            winner_latency = latency
            time_to_best = (
                time.perf_counter_ns() - started
            ) / 1_000_000.0

    if benchmark is not None and finalist_remeasurements:
        ranked = sorted(
            (
                (index, attempt)
                for index, attempt in enumerate(attempts)
                if attempt.latency_us is not None
            ),
            key=lambda item: item[1].latency_us,
        )[:finalist_count]
        fresh_samples: dict[int, list[float]] = {
            index: [] for index, _attempt in ranked
        }
        finalist_errors: dict[int, str] = {}
        for round_index in range(finalist_remeasurements):
            round_order = ranked if round_index % 2 == 0 else list(reversed(ranked))
            for index, attempt in round_order:
                if index in finalist_errors:
                    continue
                candidate = attempt.candidate
                program = measured_programs[candidate.candidate_id]
                try:
                    latency = float(benchmark(program, candidate))
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError(
                            "finalist benchmark returned a non-positive latency"
                        )
                    fresh_samples[index].append(latency)
                except Exception as error:
                    finalist_errors[index] = f"{type(error).__name__}: {error}"

        for index, attempt in ranked:
            candidate = attempt.candidate
            if index in finalist_errors:
                attempts[index] = replace(
                    attempt,
                    reason=(
                        attempt.reason
                        + "; finalist remeasurement failed: "
                        + finalist_errors[index]
                    ),
                )
                continue
            samples = fresh_samples[index]
            if not samples:
                continue
            robust_latency = float(statistics.median(samples))
            attempts[index] = replace(
                attempt,
                latency_us=robust_latency,
                reason=(
                    "performance-only selection over admitted candidate; "
                    f"finalist median of {len(samples)} fresh round-robin "
                    "measurements"
                ),
                cache_hit=False,
            )
            performance_cache.store(
                cache_keys[candidate.candidate_id],
                robust_latency,
                candidate=candidate,
            )

        measured = [
            attempt for attempt in attempts if attempt.latency_us is not None
        ]
        if measured:
            robust_winner = min(measured, key=lambda item: item.latency_us)
            winner = robust_winner.candidate
            winner_latency = robust_winner.latency_us
            winner_program = measured_programs[winner.candidate_id]

    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    winner_state = (
        concrete_states.get(winner.candidate_id)
        if winner is not None
        else None
    )
    backend_closure: BackendClosureReceipt | None = None
    backend_closure_error: str | None = None
    if (
        qd_controller is not None
        and winner_program is not None
        and winner_state is not None
    ):
        try:
            backend_closure = qd_controller.close_winner(
                winner_program,
                winner_state,
            )
            if not backend_closure.passed:
                backend_closure_error = (
                    "fixed winner failed a direct edge on held-out D_cert"
                )
        except BackendClosureError as error:
            backend_closure_error = f"{type(error).__name__}: {error}"
    if backend_closure is not None and not backend_closure.passed:
        raise ValueError(
            "fixed CUDA winner failed held-out D_cert; D_cert is not reused "
            "to select a replacement candidate"
        )
    if require_backend_closure and (
        backend_closure is None or not backend_closure.passed
    ):
        raise ValueError(
            "fixed CUDA winner lacks backend closure: "
            + (backend_closure_error or "no executable winner")
        )
    return CudaSearchResult(
        state=state,
        contract_view=contract_view,
        attempts=tuple(attempts),
        winner=winner,
        winner_program=winner_program,
        winner_latency_us=winner_latency,
        toolchain_digest=toolchain,
        benchmark_key=benchmark_key,
        elapsed_ms=elapsed,
        time_to_best_ms=time_to_best,
        finalist_count=finalist_count if benchmark is not None else 0,
        finalist_remeasurements=(
            finalist_remeasurements if benchmark is not None else 0
        ),
        b_transformations_enabled=enable_b_transformations,
        winner_state=winner_state,
        backend_closure=backend_closure,
        backend_closure_error=backend_closure_error,
        ablation_axes=axes,
        fixed_lowering_path_kind=fixed_lowering_path_kind,
        fixed_fast_path_id=fixed_fast_path_id,
        hard_admission=hard_admission,
        candidate_universe_count=len(candidate_universe),
        candidate_budget=budget,
        budget_consumed=budget_consumed,
    )


__all__ = [
    "B_CAP_RANK0_HOST_CAPTURE",
    "B_CAP_READONLY_INPUTS",
    "B_TRANSFORM_EPILOGUE_LOAD_MOTION",
    "B_TRANSFORM_RANK0_HOST_CAPTURE",
    "B_TRANSFORM_UNIFORM_HOST_CAPTURE",
    "CACHE_PROTOCOL",
    "CONTRACT_VIEW_VERSION",
    "LINALG_FUSED_FAST_PATH_ID",
    "SEARCH_PROTOCOL",
    "CudaContractError",
    "CudaContractView",
    "CudaSearchCandidate",
    "CudaSearchResult",
    "PerformanceCache",
    "SearchAttempt",
    "admit_cuda_candidate_paths",
    "canonical_cuda_candidate",
    "compile_cuda_candidate",
    "enumerate_cuda_candidates",
    "derive_cuda_contract_view",
    "derive_default_fast_path_id",
    "derive_default_lowering_path_kind",
    "search_cuda",
    "toolchain_fingerprint",
]
