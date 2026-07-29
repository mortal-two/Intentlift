"""Contract-preserving CUDA lowering-state/schedule search over immutable M0.

Candidate legality is decided only by registered C++ whole-graph probes.  The
optional C object influences enumeration order and nothing else.  Runtime
measurements select among candidates that already passed structural admission
and compiled to cubins; measurements are never used as a correctness oracle.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, replace
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
    BackendClosureResult,
    CandidateQDResult,
    ConcreteLoweringState,
    ObservationClass,
    SearchAndCloseController,
)

from .program import analyze_j_for_cuda
from .selection import DEFAULT_PASSES, LoweringTarget, select_variant


SEARCH_PROTOCOL = "intentir.backend.cuda.measured-search.v4"
CONTRACT_VIEW_VERSION = "intentir.backend.cuda.contract-view.v2"

B_TRANSFORM_EPILOGUE_LOAD_MOTION = (
    "cuda.b.epilogue_readonly_load_motion.v1"
)
B_TRANSFORM_RANK0_HOST_CAPTURE = "cuda.b.rank0_host_value_capture.v1"
B_TRANSFORM_UNIFORM_HOST_CAPTURE = "cuda.b.uniform_host_value_capture.v1"
LINALG_FUSED_FAST_PATH_ID = "cuda.linalg.elementwise-fused.v1"
SPLITK_VARARGS_FAST_PATH_ID = "cuda.splitk.reduce-varargs.v1"
SEARCH_PARTITION_UNPARTITIONED = "unpartitioned"
SEARCH_PARTITION_SCHEDULE_SPINE = "schedule_spine"
SEARCH_PARTITION_EXPANDED = "expanded"


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
        SPLITK_VARARGS_FAST_PATH_ID: "splitk_weighted_merge",
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

    target: Mapping[str, Any]
    consumed: tuple[str, ...]
    unsupported: tuple[str, ...]
    capabilities: tuple[str, ...]
    boundary_projection: Mapping[str, Any] | None
    version: str = CONTRACT_VIEW_VERSION

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": dict(self.target),
            "supported": self.supported,
            "consumed": list(self.consumed),
            "unsupported": list(self.unsupported),
            "capabilities": list(self.capabilities),
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
        target=state.target.to_dict(),
        consumed=tuple(analysis["consumed"]),
        unsupported=tuple(analysis["unsupported"]),
        capabilities=derive_search_capabilities(state.base),
        boundary_projection=analysis["boundary_projection"],
    )
    if not view.supported:
        raise CudaContractError(
            "CUDA search cannot consume complete Layer B: "
            + ", ".join(view.unsupported)
        )
    return view


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
    search_partition: str = SEARCH_PARTITION_UNPARTITIONED

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
            "search_partition": self.search_partition,
        }

    def concrete_state(self, program: LoweredProgram) -> ConcreteLoweringState:
        record = program.transformation_record
        trace = (
            record.transformation_trace
            if record is not None
            else self.requested_trace
        )
        if not trace:
            trace = (f"cuda.{self.lowering_path_kind}.v1",)
        observation_class = (
            ObservationClass.OBSERVATION_CHANGING
            if record is not None and record.observation_changing
            else self.requested_observation_class
        )
        actual_schedule = dict(self.schedule)
        if record is not None and record.launch_plans:
            # The candidate records the requested pass options.  The concrete
            # state records what C++ actually emitted: in particular row
            # vector-width=0 is an auto request that may become native vec4.
            # Keeping these two layers distinct prevents an auto-vectorized
            # winner from being reported as a scalar implementation.
            primary = record.launch_plans[0]
            actual_schedule.update(
                {
                    "block_threads": primary.block[0],
                    "actual_grid_blocks": primary.grid[0],
                    "actual_smem_bytes": primary.smem,
                    "actual_per_thread_tile": primary.per_thread_tile,
                    "vector_width": primary.vector_width,
                    "actual_pipeline_depth": primary.pipeline_depth,
                    "actual_split_factor": primary.split_factor,
                    "actual_launch_plan_count": len(record.launch_plans),
                }
            )
        return ConcreteLoweringState(
            state_id=self.candidate_id,
            lowering_path_kind=(
                record.lowering_path_kind
                if record is not None
                else self.lowering_path_kind
            ),
            transformation_trace=tuple(trace),
            schedule=actual_schedule,
            observation_class=observation_class,
            compiler_record=program.to_record(),
        )


@dataclass(frozen=True)
class SearchAttempt:
    candidate: CudaSearchCandidate
    status: str
    reason: str
    compile_ms: float | None = None
    latency_us: float | None = None
    lowering_record: Mapping[str, Any] | None = None
    concrete_state: ConcreteLoweringState | None = None
    qd_search: CandidateQDResult | None = None
    finalist_samples_us: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "compile_ms": self.compile_ms,
            "latency_us": self.latency_us,
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
            "finalist_samples_us": list(self.finalist_samples_us),
        }


@dataclass(frozen=True)
class CudaSearchResult:
    state: BackendSearchState
    contract_view: CudaContractView
    attempts: tuple[SearchAttempt, ...]
    winner: CudaSearchCandidate | None
    winner_program: LoweredProgram | None
    winner_latency_us: float | None
    benchmark_key: str | None
    elapsed_ms: float
    time_to_best_ms: float | None
    finalist_count: int
    finalist_remeasurements: int
    b_transformations_enabled: bool
    winner_state: ConcreteLoweringState | None = None
    backend_closure: BackendClosureResult | None = None
    backend_closure_error: str | None = None
    ablation_axes: SearchAxes | None = None
    fixed_lowering_path_kind: str | None = None
    fixed_fast_path_id: str | None = None
    hard_admission: tuple[Mapping[str, Any], ...] = ()
    candidate_universe_count: int = 0
    candidate_budget: int = 0
    budget_consumed: int = 0
    c_fixed_candidate_id: str | None = None
    finalist_actual_count: int = 0
    schedule_spine_candidate_count: int = 0
    expanded_candidate_count: int = 0

    def to_record(self) -> dict[str, Any]:
        best_latency: float | None = None
        best_candidate: str | None = None
        curve: list[dict[str, Any]] = []
        consumed_budget = 0
        for attempt_index, attempt in enumerate(self.attempts, start=1):
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
        actual_records = tuple(
            record
            for item in admitted
            for record in (
                (
                    item.concrete_state.compiler_record.get(
                        "transformation_record"
                    )
                    if item.concrete_state is not None
                    else None
                ),
            )
            if isinstance(record, Mapping)
        )
        reduction_topologies = sorted(
            {
                str(record["reduction_topology"])
                for record in actual_records
                if record.get("reduction_topology") not in {None, "none"}
            }
        )
        rows_per_block_values = sorted(
            {
                int(record.get("rows_per_block", 1))
                for record in actual_records
            }
        )
        actual_fma_counts = [
            int(record.get("fma_count", 0)) for record in actual_records
        ]
        approximation_modes = sorted(
            {
                str(record.get("approximation_mode"))
                for record in actual_records
                if record.get("approximation_mode") not in {None, "none"}
            }
        )
        status_counts = {
            status: sum(item.status == status for item in self.attempts)
            for status in (
                "compiled",
                "measured",
                "rejected",
                "compile_failed",
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

        # Formal marginal gains are computed from candidates remeasured inside
        # this one search visit.  Never compare cached/initial ranking samples
        # against fresh finalist medians, and never use a separately launched E2
        # process as the primary lowering attribution for E3.
        fresh_attempts = tuple(
            item
            for item in self.attempts
            if item.latency_us is not None and item.finalist_samples_us
        )

        def best_attempt(
            pool: tuple[SearchAttempt, ...],
        ) -> SearchAttempt | None:
            return (
                min(pool, key=lambda item: item.latency_us)
                if pool
                else None
            )

        fresh_global_best = best_attempt(fresh_attempts)
        fresh_spine = tuple(
            item
            for item in fresh_attempts
            if item.candidate.search_partition
            == SEARCH_PARTITION_SCHEDULE_SPINE
        )
        fresh_expanded = tuple(
            item
            for item in fresh_attempts
            if item.candidate.search_partition == SEARCH_PARTITION_EXPANDED
        )
        spine_best = best_attempt(fresh_spine)
        expanded_only_best = best_attempt(fresh_expanded)
        schedule_spine_ablation = None
        if (
            self.ablation_axes is not None
            and self.ablation_axes.search_structural_states
            and spine_best is not None
            and fresh_global_best is not None
        ):
            assert spine_best.latency_us is not None
            assert fresh_global_best.latency_us is not None
            schedule_spine_ablation = {
                "same_run_measurement": True,
                "same_total_search_visit": True,
                "schedule_spine_candidate_count": (
                    self.schedule_spine_candidate_count
                ),
                "schedule_spine_measured_count": len(fresh_spine),
                "expanded_candidate_count": self.expanded_candidate_count,
                "expanded_state_measured_count": len(fresh_expanded),
                "schedule_spine_best_candidate_id": (
                    spine_best.candidate.candidate_id
                ),
                "schedule_spine_best_us": spine_best.latency_us,
                "schedule_spine_best_batch_us": list(
                    spine_best.finalist_samples_us
                ),
                # "expanded best" is the nested E3 global best, so it may be
                # the spine winner and the gain is guaranteed not to fall below
                # one merely because a new state loses.
                "expanded_best_candidate_id": (
                    fresh_global_best.candidate.candidate_id
                ),
                "expanded_best_us": fresh_global_best.latency_us,
                "expanded_best_batch_us": list(
                    fresh_global_best.finalist_samples_us
                ),
                "expanded_over_schedule_spine_speedup": (
                    spine_best.latency_us / fresh_global_best.latency_us
                ),
                "expanded_state_best_candidate_id": (
                    expanded_only_best.candidate.candidate_id
                    if expanded_only_best is not None
                    else None
                ),
                "expanded_state_best_us": (
                    expanded_only_best.latency_us
                    if expanded_only_best is not None
                    else None
                ),
            }

        c_fixed_schedule_ablation = None
        if (
            self.ablation_axes is not None
            and self.ablation_axes.baseline_kind is None
            and not self.ablation_axes.search_structural_states
            and self.c_fixed_candidate_id is not None
            and fresh_global_best is not None
        ):
            c_fixed_attempt = next(
                (
                    item
                    for item in fresh_attempts
                    if item.candidate.candidate_id
                    == self.c_fixed_candidate_id
                ),
                None,
            )
            if c_fixed_attempt is not None:
                assert c_fixed_attempt.latency_us is not None
                assert fresh_global_best.latency_us is not None
                c_fixed_schedule_ablation = {
                    "same_run_measurement": True,
                    "same_total_search_visit": True,
                    "c_fixed_candidate_id": self.c_fixed_candidate_id,
                    "c_fixed_us": c_fixed_attempt.latency_us,
                    "c_fixed_batch_us": list(
                        c_fixed_attempt.finalist_samples_us
                    ),
                    "schedule_only_best_candidate_id": (
                        fresh_global_best.candidate.candidate_id
                    ),
                    "schedule_only_best_us": fresh_global_best.latency_us,
                    "schedule_only_best_batch_us": list(
                        fresh_global_best.finalist_samples_us
                    ),
                    "schedule_over_c_fixed_speedup": (
                        c_fixed_attempt.latency_us
                        / fresh_global_best.latency_us
                    ),
                }

        return {
            "protocol": SEARCH_PROTOCOL,
            "materialization": self.state.base.to_record(),
            "attach_c": self.state.to_record(),
            "contract_view": self.contract_view.to_dict(),
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
            "schedule_spine_ablation": schedule_spine_ablation,
            "c_fixed_schedule_ablation": c_fixed_schedule_ablation,
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
                "count": self.finalist_actual_count,
                "requested_count": self.finalist_count,
                "actual_count": self.finalist_actual_count,
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
                "concrete_post_pass_mlir_states": sum(
                    item.concrete_state is not None for item in admitted
                ),
                "concrete_topology_states": len(
                    {
                        (
                            item.concrete_state.lowering_path_kind,
                            tuple(item.concrete_state.transformation_trace),
                        )
                        for item in admitted
                        if item.concrete_state is not None
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
                "schedule_spine_concrete_states": sum(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_SCHEDULE_SPINE
                    for item in admitted
                ),
                "expanded_concrete_states": sum(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                    for item in admitted
                ),
                "expanded_state_search_observed": any(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                    for item in admitted
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


def _candidate_id(
    lowering_path_kind: str,
    fast_path_id: str | None,
    pass_argument: str | None,
) -> str:
    compiler_path = fast_path_id or "generic"
    options = pass_argument or "default"
    return f"{lowering_path_kind}:{compiler_path}:{options}"


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


def _row_numeric_opportunities(
    state: BackendSearchState,
) -> tuple[bool, bool, int]:
    """Return A-visible numeric opportunities and last-axis reduction count."""

    payload = state.base.j.to_json()["intent"]["intent_function"]["body"]
    nodes: list[Mapping[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if isinstance(value.get("op"), str):
                nodes.append(value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    producers = {
        str(node["result"]): node
        for node in nodes
        if isinstance(node.get("result"), str)
    }

    def produced_by_mul(name: Any) -> bool:
        producer = producers.get(str(name))
        return bool(
            producer
            and producer.get("op") == "map_binary"
            and producer.get("attrs", {}).get("fn") == "mul"
        )

    has_fma = any(
        (
            node.get("op") == "reduce"
            and node.get("attrs", {}).get("fn") == "sum"
            and any(produced_by_mul(item) for item in node.get("operands", []))
        )
        or (
            node.get("op") == "map_binary"
            and node.get("attrs", {}).get("fn") == "add"
            and any(produced_by_mul(item) for item in node.get("operands", []))
        )
        for node in nodes
    )
    has_division = any(
        (
            node.get("op") == "map_binary"
            and node.get("attrs", {}).get("fn") == "div"
        )
        or (
            node.get("op") == "reduce"
            and node.get("attrs", {}).get("fn") == "mean"
        )
        for node in nodes
    )
    last_axis_reductions = sum(
        node.get("op") == "reduce"
        and node.get("attrs", {}).get("axes") == [1]
        for node in nodes
    )
    return has_fma, has_division, last_axis_reductions


def _row_materialization_opportunity(state: BackendSearchState) -> bool:
    """Prove that A exposes the exact shared-row spill/elision choice.

    The C++ lowering admits this state only when one authored rank-2 output is
    also an ancestor of the sole row reduction and of another authored output.
    Recover the same necessary graph fact from A before enumeration so kernels
    without that intermediate do not spend finite search budget on requests
    that the compiler must reject.  The C++ matcher remains the authoritative
    sufficient legality check for vector width, aliasing, and expression form.
    """

    function = state.base.j.to_json()["intent"]["intent_function"]
    raw_body = function.get("body", [])
    raw_outputs = function.get("outputs", [])
    if not isinstance(raw_body, list) or not isinstance(raw_outputs, list):
        return False
    body = [item for item in raw_body if isinstance(item, Mapping)]
    output_names = {
        str(item["name"])
        for item in raw_outputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    nodes_by_result = {
        str(item["result"]): item
        for item in body
        if isinstance(item.get("result"), str)
    }
    reductions = [
        item
        for item in body
        if item.get("op") == "reduce"
        and isinstance(item.get("attrs"), Mapping)
        and item["attrs"].get("axes") == [1]
    ]
    if len(reductions) != 1:
        return False
    reduction_operands = reductions[0].get("operands", [])
    if not isinstance(reduction_operands, list) or len(reduction_operands) != 1:
        return False
    reduction_input = reduction_operands[0]
    if not isinstance(reduction_input, str):
        return False

    def depends_on(value: str, ancestor: str, visiting: set[str]) -> bool:
        if value == ancestor:
            return True
        if value in visiting:
            return False
        node = nodes_by_result.get(value)
        if node is None:
            return False
        operands = node.get("operands", [])
        if not isinstance(operands, list):
            return False
        visiting.add(value)
        try:
            return any(
                isinstance(operand, str)
                and depends_on(operand, ancestor, visiting)
                for operand in operands
            )
        finally:
            visiting.remove(value)

    def contains_select(value: str, visiting: set[str]) -> bool:
        if value in visiting:
            return False
        node = nodes_by_result.get(value)
        if node is None:
            return False
        if node.get("op") == "select":
            return True
        operands = node.get("operands", [])
        if not isinstance(operands, list):
            return False
        visiting.add(value)
        try:
            return any(
                isinstance(operand, str) and contains_select(operand, visiting)
                for operand in operands
            )
        finally:
            visiting.remove(value)

    for output in output_names:
        producer = nodes_by_result.get(output)
        if producer is None or producer.get("op") != "map_binary":
            continue
        result_type = producer.get("type", {})
        if not isinstance(result_type, Mapping):
            continue
        shape = result_type.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        if not depends_on(reduction_input, output, set()):
            continue
        if not any(
            other != output and depends_on(other, output, set())
            for other in output_names
        ):
            continue
        if contains_select(output, set()):
            continue
        return True
    return False


def _referenced_intent_inputs(state: BackendSearchState) -> frozenset[str]:
    body = state.base.j.to_json()["intent"]["intent_function"]["body"]
    referenced: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            operands = value.get("operands")
            if isinstance(operands, list):
                referenced.update(
                    str(item) for item in operands if isinstance(item, str)
                )
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(body)
    return frozenset(referenced)


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
        # A tiny outer row domain cannot occupy the GPU even when one block uses
        # many threads.  The registered C++ lowering already switches from a
        # per-row register tile to a flat grid-stride state when N/T > 32.  Rank
        # the largest power-of-two T that still crosses that transition first:
        # for rows=2,N=4096 this is T=64, exposing many independent blocks
        # instead of only two 1024-thread blocks.  This is a target-aware Retile
        # prior over an unchanged candidate universe, not an admission rule.
        if rows < 64 and columns > 32 * 32:
            transition_threads = 1 << math.floor(
                math.log2((columns - 1) // 32)
            )
            return max(32, min(1024, transition_threads))

        # Otherwise keep roughly 64K logical threads in flight across
        # independent rows, while never assigning more lanes than the row can
        # use.  This maps 128x1024 to 512 threads and 2048x4096 to 64 threads on
        # the same target.
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
    # A fixed concrete path is the complete Schedule-Only search space.  The
    # return logic below excludes the generic materialized pipeline, which
    # would silently turn it back into a multi-path experiment.
    contract_view = derive_cuda_contract_view(state)
    capabilities = set(contract_view.capabilities)
    runtime_names = set(state.base.runtime_input_names)
    referenced_inputs = _referenced_intent_inputs(state)
    has_runtime_rank0 = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and not item.shape
        for item in state.base.j.intent.inputs
    )
    has_column_input = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and len(item.shape) == 1
        for item in state.base.j.intent.inputs
    )
    has_uniform_input = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and (not item.shape or item.shape == [1])
        for item in state.base.j.intent.inputs
    )
    priors = (
        _numeric_priors(state)
        if axes is None or axes.use_c_prior
        else ()
    )
    row_elements_per_thread = _row_elements_per_thread(state)
    has_fma_opportunity, has_division_opportunity, last_axis_reduction_count = (
        _row_numeric_opportunities(state)
    )
    has_materialization_opportunity = _row_materialization_opportunity(state)
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

    # Row vector-width=0 is the C++ auto state: it emits vec4 exactly when the
    # structural gate proves a contiguous full register tile, otherwise scalar.
    # An explicit width=4 request therefore either duplicates that exact MLIR or
    # fails admission; it is not a distinct concrete state and must not consume
    # finite search budget.  Width=1 remains the deliberate scalar ablation.
    row_schedule_bases = [
        *(RowSchedule(threads, 32, vector_width=0, fuse_fma=False)
          for threads in (32, 64, 128, 256, 512, 1024)),
        *(RowSchedule(threads, 64, vector_width=0, fuse_fma=False)
          for threads in (64, 128, 256)),
        *(RowSchedule(threads, 32, vector_width=1, fuse_fma=False)
          for threads in (64, 128, 256)),
    ]
    if axes is None or axes.search_reduction_topology:
        row_schedule_bases.extend(
            RowSchedule(32, 32, rows_per_block=rows)
            for rows in (2, 4, 8)
        )
        if last_axis_reduction_count == 1:
            row_schedule_bases.extend(
                RowSchedule(
                    threads,
                    32,
                    vector_width=1,
                    promote_rank0_scalars=False,
                    defer_epilogue_col_loads=False,
                    reduction_parts=parts,
                )
                for threads in (128, 256)
                for parts in (2, 4, 8, 16, 32)
            )
    if axes is not None and not axes.search_reduction_topology:
        # Schedule-Only keeps the concrete collective topology on the canonical
        # multi-warp hierarchy while still tuning its schedule parameters.
        row_schedule_bases = [
            item for item in row_schedule_bases if item.block_threads > 32
        ]
    fma_modes = (
        (False, True)
        if (axes is None or axes.search_fma) and has_fma_opportunity
        else (False,)
    )
    approximate_div_modes = (
        (False, True)
        if (axes is None or axes.search_precision) and has_division_opportunity
        else (False,)
    )
    leader_finalize_modes = (
        (False, True)
        if (axes is None or axes.search_reduction_topology)
        and last_axis_reduction_count > 1
        else (False,)
    )
    materialize_shared_row_output_modes = (
        (True, False)
        if (axes is None or axes.search_structural_states)
        and has_materialization_opportunity
        else (True,)
    )
    promote_rank0_modes = (True, False) if has_runtime_rank0 else (False,)
    defer_column_modes = (True, False) if has_column_input else (False,)
    row_schedules = [
        replace(
            schedule,
            fuse_fma=fuse_fma,
            approximate_div=approximate_div,
            leader_finalize=leader_finalize,
            promote_rank0_scalars=promote,
            defer_epilogue_col_loads=defer,
            materialize_shared_row_output=materialize_row_output,
        )
        for schedule in row_schedule_bases
        for fuse_fma in fma_modes
        for approximate_div in approximate_div_modes
        for leader_finalize in leader_finalize_modes
        for promote in promote_rank0_modes
        for defer in defer_column_modes
        for materialize_row_output in materialize_shared_row_output_modes
        if schedule.reduction_parts == 1
        and (
            not leader_finalize
            or (schedule.block_threads > 32 and schedule.rows_per_block == 1)
        )
    ]
    row_schedules.extend(
        schedule for schedule in row_schedule_bases if schedule.reduction_parts > 1
    )
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
                        "cuda.reduction.multi_stage_partial_finalize.v1"
                        if schedule.reduction_parts > 1
                        else "cuda.reduction.subgroup_rows.v1"
                        if schedule.rows_per_block > 1
                        else "cuda.reduction.workgroup_leader_broadcast.v1"
                        if schedule.leader_finalize
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
                    *(
                        ("cuda.elide.intermediate_row_materialization.v1",)
                        if not schedule.materialize_shared_row_output
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
    # is only admitted when IntentIR searches topology; Schedule-Only remains
    # on the fixed serial-row state.  The C++ probe supports a bounded strided
    # feature strip
    # (including natural head dimensions 64/128/256) and remains the hard gate.
    if axes is None or axes.search_reduction_topology:
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                1,
                lanes_per_row,
            )
            for threads in (64, 128, 256, 512, 1024)
            for vectorize in (False, True)
            for lanes_per_row in (32,)
        )
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                1,
                lanes_per_row,
            )
            for threads in (128, 256)
            for vectorize in (False, True)
            for lanes_per_row in (4, 8, 16)
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
                    *(
                        ("cuda.reduction.padded_subgroup_multirow.v1",)
                        if schedule.rows_per_subgroup > 1
                        else ()
                    ),
                    *transformations,
                ),
                ObservationClass.OBSERVATION_CHANGING,
            )
        )

    # The typed unrolled SplitK graph has one concrete whole-graph compiler path
    # and a genuine launch-schedule axis: the number of output rows assigned to
    # each warp block.  Keep the pass default (rows=4) byte-identical to the
    # canonical compiler state so the Schedule-Only prefix is nested, and
    # enumerate the remaining legal C++ options as ordinary schedule states.
    # The compiler probe still decides
    # whether the current J has the required weighted-merge SSA topology.
    for rows_per_block in (4, 1, 2, 8):
        pass_argument = (
            "-convert-intent-splitk-reduce-varargs-to-gpu"
            if rows_per_block == 4
            else (
                "-convert-intent-splitk-reduce-varargs-to-gpu="
                f"rows-per-block={rows_per_block}"
            )
        )
        values = {
            "rows_per_block": rows_per_block,
            "block_threads": rows_per_block * 32,
        }
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "splitk_weighted_merge",
                    SPLITK_VARARGS_FAST_PATH_ID,
                    pass_argument,
                ),
                "splitk_weighted_merge",
                SPLITK_VARARGS_FAST_PATH_ID,
                pass_argument,
                values,
                _c_score(values, priors),
                requested_trace=(
                    "cuda.splitk.unrolled_weighted_merge.v2",
                    "cuda.materialization.whole_graph_elimination.v1",
                ),
                requested_observation_class=(
                    ObservationClass.OBSERVATION_CHANGING
                ),
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
        SPLITK_VARARGS_FAST_PATH_ID,
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
    # another. Use C only to order schedules inside each registered path kind;
    # expanded arms later prepend the fixed-structure schedule spine before
    # round-robin path discovery so IntentIR cannot forget Schedule-Only under
    # a finite budget.
    def search_lane(item: CudaSearchCandidate) -> str:
        local = "local" if item.b_transformations else "base"
        if item.lowering_path_kind == "row_reduction":
            topology = (
                "multi_stage"
                if int(item.schedule.get("reduction_parts", 1)) > 1
                else "subgroup_rows"
                if int(item.schedule.get("rows_per_block", 1)) > 1
                else "workgroup_leader"
                if item.schedule.get("leader_finalize") is True
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
            if item.schedule.get("materialize_shared_row_output") is False:
                return f"row_reduction.{topology}.{numeric}.elide.{local}"
            return f"row_reduction.{topology}.{numeric}.{local}"
        if item.lowering_path_kind == "elementwise":
            approximation = str(
                item.schedule.get("tanh_approximation", "stable-ex2")
            )
            return f"elementwise.{approximation}.{local}"
        if item.lowering_path_kind == "padded_row_reduction":
            topology = str(item.schedule.get("reduction_topology", "serial-row"))
            packet = (
                "packet"
                if item.schedule.get("vectorize_full_features") is True
                else "scalar"
            )
            lanes = int(item.schedule.get("lanes_per_row", 32))
            return (
                f"{item.lowering_path_kind}.{topology}.{packet}."
                f"lanes{lanes}.{local}"
            )
        return item.lowering_path_kind

    known_lane_order = (
        "elementwise.stable-ex2.base",
        "row_reduction.workgroup.plain.base",
        "padded_row_reduction.serial-row.scalar.lanes32.base",
        "padded_row_reduction.serial-row.packet.lanes32.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes32.base",
        "padded_row_reduction.subgroup-feature.packet.lanes16.base",
        "padded_row_reduction.subgroup-feature.packet.lanes8.base",
        "padded_row_reduction.subgroup-feature.packet.lanes4.base",
        "padded_row_reduction.subgroup-feature.packet.lanes32.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes16.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes8.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes4.base",
        "row_reduction.multi_stage.plain.base",
        "elementwise.native-ptx.base",
        "row_reduction.subgroup_rows.plain.base",
        "row_reduction.subgroup.plain.base",
        "row_reduction.workgroup.fma.base",
        "row_reduction.subgroup.fma.base",
        "row_reduction.workgroup.approximate_div.base",
        "row_reduction.subgroup_rows.approximate_div.base",
        "row_reduction.subgroup.approximate_div.base",
        "row_reduction.workgroup_leader.plain.base",
        "row_reduction.workgroup_leader.fma.base",
        "row_reduction.workgroup_leader.approximate_div.base",
        "row_reduction.workgroup.plain.elide.base",
        "row_reduction.workgroup.fma.elide.base",
        "row_reduction.workgroup.approximate_div.elide.base",
        "row_reduction.workgroup_leader.plain.elide.base",
        "row_reduction.workgroup_leader.fma.elide.base",
        "row_reduction.workgroup_leader.approximate_div.elide.base",
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
        "row_reduction.workgroup.plain.elide.local",
        "row_reduction.workgroup.fma.elide.local",
        "row_reduction.workgroup.approximate_div.elide.local",
        "padded_row_reduction.serial-row.scalar.lanes32.local",
        "padded_row_reduction.serial-row.packet.lanes32.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes32.local",
        "padded_row_reduction.subgroup-feature.packet.lanes16.local",
        "padded_row_reduction.subgroup-feature.packet.lanes8.local",
        "padded_row_reduction.subgroup-feature.packet.lanes4.local",
        "padded_row_reduction.subgroup-feature.packet.lanes32.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes16.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes8.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes4.local",
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
    expanded = (generic, *specialized)
    if axes is None or not axes.search_structural_states:
        return expanded

    # IntentIR expands the concrete state space, but its fixed budget must retain
    # the complete Schedule-Only prefix.  Build that prefix through this exact
    # enumerator with structural/numeric/local expansion disabled.
    canonical = canonical_cuda_candidate(state)
    schedule_axes = replace(
        axes,
        search_structural_states=False,
        enable_local_b_transformations=False,
        search_reduction_topology=False,
        search_fma=False,
        search_precision=False,
        search_approximation=False,
    )
    schedule_spine = enumerate_cuda_candidates(
        state,
        enable_b_transformations=False,
        axes=schedule_axes,
        fixed_lowering_path_kind=(
            "generic" if canonical.generic else canonical.lowering_path_kind
        ),
        fixed_fast_path_id=(None if canonical.generic else canonical.fast_path_id),
    )
    spine_ids = {item.candidate_id for item in schedule_spine}
    expanded_remainder = [
        item for item in expanded if item.candidate_id not in spine_ids
    ]

    def schedule_coordinate(
        item: CudaSearchCandidate,
    ) -> tuple[object, ...] | None:
        if item.lowering_path_kind == "row_reduction":
            return (
                item.schedule.get("block_threads"),
                item.schedule.get("tile_cols"),
                item.schedule.get("vector_width"),
            )
        if item.lowering_path_kind == "elementwise":
            return (
                item.schedule.get("block_threads"),
                item.schedule.get("flatten_contiguous"),
                item.schedule.get("vector_width"),
            )
        return None

    def same_path_lowering_choice(
        item: CudaSearchCandidate,
    ) -> tuple[object, ...] | None:
        if (
            item.fast_path_id != canonical.fast_path_id
            or item.b_transformations
            or schedule_coordinate(item) is None
        ):
            return None
        if item.lowering_path_kind == "row_reduction":
            # Topology-changing states own different schedule domains and stay
            # in the ordinary lowering-path order below.  Here we cross the
            # same workgroup schedules with A-visible numeric evaluation and
            # intermediate-materialization choices.
            if (
                int(item.schedule.get("rows_per_block", 1)) != 1
                or int(item.schedule.get("reduction_parts", 1)) != 1
                or int(item.schedule.get("block_threads", 64)) == 32
            ):
                return None
            choice = (
                bool(item.schedule.get("fuse_fma", False)),
                bool(item.schedule.get("approximate_div", False)),
                bool(item.schedule.get("leader_finalize", False)),
                item.schedule.get("materialize_shared_row_output", True) is False,
            )
            return ("row_reduction", *choice) if any(choice) else None
        if item.lowering_path_kind == "elementwise":
            approximation = str(
                item.schedule.get("tanh_approximation", "stable-ex2")
            )
            if approximation != "stable-ex2":
                return ("elementwise", approximation)
        return None

    # A finite IntentIR search must exercise each admitted same-path lowering
    # choice at more than one schedule without crowding out other paths.  Take
    # the first three distinct thread counts in the C-ranked Schedule-Only
    # order and cross both individual changes and their interactions with all
    # three.  A's opportunity gates make materialization mutually exclusive
    # with multi-reduction finalization, so at most three independent axes are
    # active here and the complete three-schedule cross fits budget 32 after the
    # fixed Schedule-Only spine.  Remaining lowering paths retain their ordinary
    # path-fair order afterwards.
    by_lowering_choice: dict[
        tuple[object, ...], list[CudaSearchCandidate]
    ] = {}
    for item in expanded_remainder:
        choice = same_path_lowering_choice(item)
        if choice is not None:
            by_lowering_choice.setdefault(choice, []).append(item)

    single_choices = [
        choice
        for choice in by_lowering_choice
        if choice[0] != "row_reduction"
        or sum(bool(item) for item in choice[1:]) == 1
    ]
    interaction_choices = [
        choice
        for choice in by_lowering_choice
        if choice[0] == "row_reduction"
        and sum(bool(item) for item in choice[1:]) > 1
    ]
    representative_spine: list[CudaSearchCandidate] = []
    observed_thread_counts: set[object] = set()
    for item in schedule_spine:
        block_threads = item.schedule.get("block_threads")
        if block_threads in observed_thread_counts:
            continue
        observed_thread_counts.add(block_threads)
        representative_spine.append(item)
        if len(representative_spine) == 3:
            break

    prioritized: list[CudaSearchCandidate] = []
    prioritized_ids: set[str] = set()
    for spine_item in representative_spine:
        coordinate = schedule_coordinate(spine_item)
        if coordinate is None:
            continue
        choices = (*single_choices, *interaction_choices)
        for choice in choices:
            candidates = by_lowering_choice[choice]
            match = next(
                (
                    item
                    for item in candidates
                    if schedule_coordinate(item) == coordinate
                    and item.candidate_id not in prioritized_ids
                ),
                None,
            )
            if match is not None:
                prioritized.append(match)
                prioritized_ids.add(match.candidate_id)

    ordered_expanded = (
        *prioritized,
        *(
            item
            for item in expanded_remainder
            if item.candidate_id not in prioritized_ids
        ),
    )

    return (
        *(
            replace(
                item,
                search_partition=SEARCH_PARTITION_SCHEDULE_SPINE,
            )
            for item in schedule_spine
        ),
        *(
            replace(item, search_partition=SEARCH_PARTITION_EXPANDED)
            for item in ordered_expanded
        ),
    )


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
        raise ValueError("CUDA search budget differs from configured ablation arm")
    if axes is not None:
        enable_b_transformations = axes.enable_local_b_transformations
    if require_backend_closure and qd_controller is None:
        raise ValueError("formal backend closure requires SearchAndClose")
    started = time.perf_counter_ns()
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
            # registered C++ matcher admits the current IR.  Keep Schedule-Only
            # on the canonical generic state instead of manufacturing a missing
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
    # Budget counts the explicit, directly readable candidate list.
    candidates = list(admitted_candidates)
    c_fixed_candidate_id = (
        candidates[0].candidate_id if candidates else None
    )
    schedule_spine_candidate_count = sum(
        item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
        for item in candidates
    )
    expanded_candidate_count = sum(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        for item in candidates
    )

    attempts: list[SearchAttempt] = []
    winner: CudaSearchCandidate | None = None
    winner_program: LoweredProgram | None = None
    winner_latency: float | None = None
    time_to_best: float | None = None
    measured_programs: dict[str, LoweredProgram] = {}
    concrete_states: dict[str, ConcreteLoweringState] = {}
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
        except Exception as error:  # compiler diagnostics belong in the attempt
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
        status = "compiled"
        reason = (
            "complete Layer-B audit, C++ whole-graph admission, and cubin "
            "compilation passed"
        )
        if benchmark is not None:
            assert benchmark_key is not None
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

    finalist_actual_count = 0
    if benchmark is not None and finalist_remeasurements:
        initial_ranked = sorted(
            (
                (index, attempt)
                for index, attempt in enumerate(attempts)
                if attempt.latency_us is not None
            ),
            key=lambda item: item[1].latency_us,
        )

        def provisional_best_index(
            predicate: Callable[[SearchAttempt], bool],
        ) -> int | None:
            eligible = [
                (index, attempt)
                for index, attempt in initial_ranked
                if predicate(attempt)
            ]
            return eligible[0][0] if eligible else None

        # Preserve the causal contrasts even when a small finalist_count would
        # otherwise exclude a slow C projection or one side of the nested
        # Schedule-Only/IntentIR comparison.
        mandatory_indices = [
            index
            for index in (
                provisional_best_index(
                    lambda item: item.candidate.candidate_id
                    == c_fixed_candidate_id
                ),
                provisional_best_index(
                    lambda item: item.candidate.search_partition
                    == SEARCH_PARTITION_SCHEDULE_SPINE
                ),
                provisional_best_index(
                    lambda item: item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                ),
            )
            if index is not None
        ]
        ranked_indices: list[int] = []
        for index in (
            *mandatory_indices,
            *(index for index, _attempt in initial_ranked),
        ):
            if index not in ranked_indices:
                ranked_indices.append(index)
            if (
                len(ranked_indices) >= finalist_count
                and all(item in ranked_indices for item in mandatory_indices)
            ):
                break
        ranked = [(index, attempts[index]) for index in ranked_indices]
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
                finalist_samples_us=tuple(samples),
            )

        freshly_remeasured = [
            attempts[index]
            for index, _attempt in ranked
            if attempts[index].finalist_samples_us
        ]
        finalist_actual_count = len(freshly_remeasured)
        if freshly_remeasured:
            robust_winner = min(
                freshly_remeasured,
                key=lambda item: item.latency_us,
            )
            winner = robust_winner.candidate
            winner_latency = robust_winner.latency_us
            winner_program = measured_programs[winner.candidate_id]

    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    winner_state = (
        concrete_states.get(winner.candidate_id)
        if winner is not None
        else None
    )
    backend_closure: BackendClosureResult | None = None
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
        c_fixed_candidate_id=c_fixed_candidate_id,
        finalist_actual_count=finalist_actual_count,
        schedule_spine_candidate_count=schedule_spine_candidate_count,
        expanded_candidate_count=expanded_candidate_count,
    )


__all__ = [
    "B_CAP_RANK0_HOST_CAPTURE",
    "B_CAP_READONLY_INPUTS",
    "B_TRANSFORM_EPILOGUE_LOAD_MOTION",
    "B_TRANSFORM_RANK0_HOST_CAPTURE",
    "B_TRANSFORM_UNIFORM_HOST_CAPTURE",
    "CONTRACT_VIEW_VERSION",
    "LINALG_FUSED_FAST_PATH_ID",
    "SEARCH_PROTOCOL",
    "CudaContractError",
    "CudaContractView",
    "CudaSearchCandidate",
    "CudaSearchResult",
    "SEARCH_PARTITION_EXPANDED",
    "SEARCH_PARTITION_SCHEDULE_SPINE",
    "SEARCH_PARTITION_UNPARTITIONED",
    "SearchAttempt",
    "admit_cuda_candidate_paths",
    "canonical_cuda_candidate",
    "compile_cuda_candidate",
    "enumerate_cuda_candidates",
    "derive_cuda_contract_view",
    "derive_default_fast_path_id",
    "derive_default_lowering_path_kind",
    "search_cuda",
]
