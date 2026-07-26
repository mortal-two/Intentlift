"""Contract-preserving CUDA family/schedule search over immutable M0.

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
from intentir.backend.cuda.pipeline import LoweredProgram, lower_to_cuda
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
from intentir.backend.materialize import BackendSearchState
from intentir.ir.production_snapshot import content_digest

from .program import analyze_j_for_cuda


SEARCH_PROTOCOL = "intentir.backend.cuda.measured-search.v2"
CACHE_PROTOCOL = "intentir.backend.cuda.performance-cache.v1"
CONTRACT_VIEW_VERSION = "intentir.backend.cuda.contract-view.v2"

B_TRANSFORM_EPILOGUE_LOAD_MOTION = (
    "cuda.b.epilogue_readonly_load_motion.v1"
)
B_TRANSFORM_RANK0_HOST_CAPTURE = "cuda.b.rank0_host_value_capture.v1"
B_TRANSFORM_UNIFORM_HOST_CAPTURE = "cuda.b.uniform_host_value_capture.v1"


Benchmark = Callable[[LoweredProgram, "CudaSearchCandidate"], float]


class CudaContractError(ValueError):
    """Raised before candidate compilation when complete Layer B is unsupported."""


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
    family: str
    fast_path_id: str | None
    pass_argument: str | None
    schedule: Mapping[str, int | bool | str]
    c_order_score: float
    b_transformations: tuple[str, ...] = ()
    required_b_capabilities: tuple[str, ...] = ()

    @property
    def generic(self) -> bool:
        return self.fast_path_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "schedule": dict(self.schedule),
            "c_order_score": self.c_order_score,
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(
                self.required_b_capabilities
            ),
        }


@dataclass(frozen=True)
class SearchAttempt:
    candidate: CudaSearchCandidate
    status: str
    reason: str
    compile_ms: float | None = None
    latency_us: float | None = None
    cache_hit: bool = False
    lowering_record: Mapping[str, Any] | None = None

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

    def to_record(self) -> dict[str, Any]:
        best_latency: float | None = None
        best_candidate: str | None = None
        curve: list[dict[str, Any]] = []
        for attempt_index, attempt in enumerate(self.attempts, start=1):
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
                    "attempt_budget": attempt_index,
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

        return {
            "protocol": SEARCH_PROTOCOL,
            "materialization": self.state.base.to_record(),
            "attach_c": self.state.to_record(),
            "contract_view": self.contract_view.to_dict(),
            "toolchain_digest": self.toolchain_digest,
            "benchmark_key": self.benchmark_key,
            "attempts": [item.to_dict() for item in self.attempts],
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "winner_latency_us": self.winner_latency_us,
            "elapsed_ms": self.elapsed_ms,
            "time_to_best_ms": self.time_to_best_ms,
            "finalist_tournament": {
                "count": self.finalist_count,
                "remeasurements_per_candidate": self.finalist_remeasurements,
                "aggregation": "median including initial measurement",
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
            "budget_curve": curve,
            "counts": {
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
                    "family": candidate.family,
                    "fast_path_id": candidate.fast_path_id,
                    "pass_argument": candidate.pass_argument,
                    "schedule": dict(candidate.schedule),
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
    family: str,
    fast_path_id: str | None,
    pass_argument: str | None,
) -> str:
    suffix = content_digest(
        {
            "family": family,
            "fast_path_id": fast_path_id,
            "pass_argument": pass_argument,
        }
    )[:12]
    return f"{family}.{suffix}"


def _numeric_priors(state: BackendSearchState) -> tuple[float, ...]:
    return tuple(
        float(item.value)
        for item in state.priors
        if item.role in {"tile_extent", "reduction_tile_extent"}
        and isinstance(item.value, (int, float))
        and not isinstance(item.value, bool)
        and item.value > 0
    )


def _c_score(
    schedule: Mapping[str, int | bool | str],
    priors: tuple[float, ...],
) -> float:
    if not priors:
        return 0.0
    values = [
        float(schedule[name])
        for name in ("block_threads", "tile_cols", "tile_rows")
        if isinstance(schedule.get(name), int)
        and not isinstance(schedule.get(name), bool)
        and int(schedule[name]) > 0
    ]
    if not values:
        return float("inf")
    return min(
        abs(math.log2(value / prior))
        for value in values
        for prior in priors
    )


def enumerate_cuda_candidates(
    state: BackendSearchState,
    *,
    enable_b_transformations: bool = True,
) -> tuple[CudaSearchCandidate, ...]:
    """Enumerate families only after complete B-conditioned admission."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA candidate enumeration requires BackendSearchState")
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
    priors = _numeric_priors(state)
    specialized: list[CudaSearchCandidate] = []

    elementwise_schedule_bases = [
        *(ElementwiseSchedule(threads, False, 1)
          for threads in (32, 64, 128, 256, 512)),
        *(ElementwiseSchedule(threads, True, width)
          for threads in (64, 128, 256)
          for width in (2, 4)),
    ]
    elementwise_schedules = [
        replace(base, promote_uniform_scalars=promote)
        for base in elementwise_schedule_bases
        for promote in ((False, True) if has_uniform_input else (False,))
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
                _c_score(values, priors),
                tuple(transformations),
                tuple(requirements),
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
    row_schedules = [
        replace(
            schedule,
            promote_rank0_scalars=promote,
            defer_epilogue_col_loads=defer,
        )
        for schedule in row_schedule_bases
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
                _c_score(values, priors),
                tuple(transformations),
                tuple(requirements),
            )
        )

    padded_schedule_bases = (
        PaddedRowSchedule(32, False),
        PaddedRowSchedule(64, False),
        PaddedRowSchedule(128, False),
        PaddedRowSchedule(32, True),
        PaddedRowSchedule(64, True),
        PaddedRowSchedule(128, True),
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
            )
        )

    # A C tile/thread prior has no authority to rank one implementation family
    # above another.  Keep family discovery fair under a finite budget and use
    # C only to order schedules *inside* each registered family.
    family_order = (
        "elementwise",
        "row_reduction",
        "padded_row_reduction",
    )
    by_family = {
        family: [item for item in specialized if item.family == family]
        for family in family_order
    }
    for candidates in by_family.values():
        candidates.sort(key=lambda item: item.c_order_score)
    specialized = []
    rank = 0
    while any(rank < len(by_family[family]) for family in family_order):
        for family in family_order:
            candidates = by_family[family]
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
    )
    return (generic, *specialized)


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
    started = time.perf_counter_ns()
    toolchain = toolchain_fingerprint()
    performance_cache = cache or PerformanceCache()
    contract_view = derive_cuda_contract_view(state)
    planned = enumerate_cuda_candidates(
        state,
        enable_b_transformations=enable_b_transformations,
    )
    generic = planned[0]
    candidates = [*(([generic] if include_generic else [])), *planned[1 : budget + 1]]

    attempts: list[SearchAttempt] = []
    winner: CudaSearchCandidate | None = None
    winner_program: LoweredProgram | None = None
    winner_latency: float | None = None
    time_to_best: float | None = None
    measured_programs: dict[str, LoweredProgram] = {}
    cache_keys: dict[str, str] = {}

    for candidate in candidates:
        compile_started = time.perf_counter_ns()
        try:
            if candidate.generic:
                program = lower_to_cuda(
                    state.base_ir,
                    allow_specialized=False,
                )
            else:
                assert candidate.fast_path_id is not None
                assert candidate.pass_argument is not None
                program = lower_to_cuda(
                    state.base_ir,
                    only_fast_path=candidate.fast_path_id,
                    pass_overrides={
                        candidate.fast_path_id: candidate.pass_argument
                    },
                    require_specialized=True,
                )
        except ValueError as error:
            attempts.append(
                SearchAttempt(
                    candidate,
                    "rejected",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        except Exception as error:  # compiler diagnostics belong in the receipt
            attempts.append(
                SearchAttempt(
                    candidate,
                    "compile_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0

        latency: float | None = None
        cache_hit = False
        status = "compiled"
        reason = (
            "complete Layer-B audit, C++ whole-graph admission, and cubin "
            "compilation passed"
        )
        if benchmark is not None and not candidate.generic:
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
        for index, attempt in ranked:
            candidate = attempt.candidate
            program = measured_programs[candidate.candidate_id]
            samples = [float(attempt.latency_us)]
            try:
                for _ in range(finalist_remeasurements):
                    latency = float(benchmark(program, candidate))
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError(
                            "finalist benchmark returned a non-positive latency"
                        )
                    samples.append(latency)
            except Exception as error:
                attempts[index] = replace(
                    attempt,
                    reason=(
                        attempt.reason
                        + "; finalist remeasurement failed: "
                        + f"{type(error).__name__}: {error}"
                    ),
                )
                continue
            robust_latency = float(statistics.median(samples))
            attempts[index] = replace(
                attempt,
                latency_us=robust_latency,
                reason=(
                    "performance-only selection over admitted candidate; "
                    f"finalist median of {len(samples)} measurements"
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
    )


__all__ = [
    "B_CAP_RANK0_HOST_CAPTURE",
    "B_CAP_READONLY_INPUTS",
    "B_TRANSFORM_EPILOGUE_LOAD_MOTION",
    "B_TRANSFORM_RANK0_HOST_CAPTURE",
    "B_TRANSFORM_UNIFORM_HOST_CAPTURE",
    "CACHE_PROTOCOL",
    "CONTRACT_VIEW_VERSION",
    "SEARCH_PROTOCOL",
    "CudaContractError",
    "CudaContractView",
    "CudaSearchCandidate",
    "CudaSearchResult",
    "PerformanceCache",
    "SearchAttempt",
    "enumerate_cuda_candidates",
    "derive_cuda_contract_view",
    "search_cuda",
    "toolchain_fingerprint",
]
