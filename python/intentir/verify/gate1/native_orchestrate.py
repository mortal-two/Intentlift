"""Gate1 runners for CUDA C++ and TileLang public frontends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from .cuda_compiler import (
    CudaCompileCase,
    CudaCompileResult,
    CudaCompiler,
    CudaKernelSource,
    FrontendParameter,
)
from .model import Gate1Artifact, Gate1Status, ParameterKind
from .orchestrate import (
    Gate1ArmEvidence,
    Gate1Config,
    Gate1ExecutionResult,
    canonical_artifact_digest,
)
from .profile import CUDA_OBSERVER_PROFILE, TILELANG_OBSERVER_PROFILE
from .ptx import PTXComparator, PTXFingerprint, fingerprint_ptx
from .replay import ReplayWriter
from .tilelang import TileLangComparator, TileLangFingerprint, fingerprint_tilelang
from .tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompileResult,
    TileLangCompiler,
    TileLangKernelSource,
)


def _dtype(value: str | None) -> str:
    aliases = {
        "bool": "i1",
        "float": "f32",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "half": "f16",
        "int": "i32",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    key = str(value or "").removeprefix("torch.").lower()
    return aliases.get(key, key)


def _artifact_interface(
    artifact: Gate1Artifact,
    parameters: tuple[FrontendParameter, ...],
) -> dict[str, Any]:
    signature = artifact.j.signature.parameters
    if len(signature) != len(parameters):
        raise ValueError("frontend Harness parameter count differs from J.signature")
    expected_kind = {
        ParameterKind.POINTER: "pointer",
        ParameterKind.TENSOR: "pointer",
        ParameterKind.SCALAR: "scalar",
        ParameterKind.OPAQUE: "opaque",
    }
    normalized: list[dict[str, Any]] = []
    for index, (semantic, harness) in enumerate(zip(signature, parameters, strict=True)):
        if semantic.kind is ParameterKind.CONSTEXPR:
            raise ValueError(
                "CUDA/TileLang runtime ABI may not expose J constexpr parameters; use C"
            )
        kind = expected_kind[semantic.kind]
        if harness.kind != kind:
            raise ValueError(f"frontend parameter {index} kind differs from J.signature")
        if _dtype(semantic.dtype) != _dtype(harness.dtype):
            raise ValueError(f"frontend parameter {index} dtype differs from J.signature")
        if semantic.rank is not None and semantic.rank != harness.rank:
            raise ValueError(f"frontend parameter {index} rank differs from J.signature")
        if semantic.mode.value != harness.mode:
            raise ValueError(f"frontend parameter {index} mode differs from J.signature")
        normalized.append(
            {
                "position": index,
                "kind": kind,
                "dtype": _dtype(semantic.dtype),
                "rank": semantic.rank,
                "mode": semantic.mode.value,
            }
        )
    return {"parameters": normalized}


def _validate_schedule(artifact: Gate1Artifact, schedule: Mapping[str, Any]) -> None:
    if artifact.c.values != dict(schedule):
        raise ValueError("frontend Harness schedule differs from the accepted C")


def _replay_writers(
    writer: ReplayWriter | tuple[ReplayWriter, ...],
) -> tuple[ReplayWriter, ...]:
    writers = writer if isinstance(writer, tuple) else (writer,)
    if not writers:
        raise ValueError("Gate1 requires at least one Replay writer")
    if len({id(item) for item in writers}) != len(writers):
        raise ValueError("Gate1 Replay fan-out requires fresh independent writers")
    return writers


def _normal_arms(
    runner: Any,
    artifact: Gate1Artifact,
    source_fingerprint: PTXFingerprint | TileLangFingerprint,
    case: CudaCompileCase | TileLangCompileCase,
) -> tuple[Gate1ArmEvidence, ...]:
    if len(runner.writers) == 1:
        return (
            runner._arm(
                runner.writers[0],
                artifact,
                source_fingerprint,
                case,
                masked=False,
            ),
        )
    with ThreadPoolExecutor(max_workers=len(runner.writers)) as executor:
        futures = [
            executor.submit(
                runner._arm,
                writer,
                artifact,
                source_fingerprint,
                case,
                masked=False,
            )
            for writer in runner.writers
        ]
        return tuple(future.result() for future in futures)


def _selected_normal_index(candidates: tuple[Gate1ArmEvidence, ...]) -> int:
    return next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.hard_closed
        ),
        0,
    )


class CudaGate1Runner:
    def __init__(
        self,
        *,
        compiler: CudaCompiler,
        writer: ReplayWriter | tuple[ReplayWriter, ...],
        masked_writer: ReplayWriter | None = None,
        comparator: PTXComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.config = config or Gate1Config()
        self.writers = _replay_writers(writer)
        if self.config.run_masked_j:
            if masked_writer is None:
                raise ValueError("masked-J requires a fresh independent Replay writer")
            if id(masked_writer) in {id(item) for item in self.writers}:
                raise ValueError("masked-J writer must differ from every normal writer")
        elif masked_writer is not None:
            raise ValueError("masked-J writer was provided while the audit is disabled")
        self.writer = self.writers[0]
        self.masked_writer = masked_writer
        self.comparator = comparator or PTXComparator()

    def _fingerprint(
        self,
        artifact: Gate1Artifact,
        result: CudaCompileResult,
        case: CudaCompileCase,
    ) -> PTXFingerprint:
        if result.ptx is None or result.entry is None:
            raise ValueError("CUDA compile result lacks PTX/entry")
        interface = _artifact_interface(artifact, case.parameters)
        return fingerprint_ptx(
            result.ptx,
            entry=result.entry,
            interface=interface,
            target={"arch": case.arch},
            schedule=case.schedule,
        )

    def _arm(
        self,
        writer: ReplayWriter,
        artifact: Gate1Artifact,
        source_fingerprint: PTXFingerprint,
        case: CudaCompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = writer.generate(artifact, masked=masked)
        except Exception as exc:
            attempt = getattr(exc, "attempt", None)
            return Gate1ArmEvidence(
                mode,
                None,
                None,
                None,
                None,
                f"generation failed: {type(exc).__name__}: {exc}",
                getattr(attempt, "attempts", 0),
            )
        try:
            replay_source = CudaKernelSource.from_replay(
                generation.code, generation.function_name
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                None,
                None,
                None,
                f"Replay policy failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        compiled = self.compiler.compile(replay_source, case)
        if not compiled.ok:
            return Gate1ArmEvidence(
                mode,
                generation,
                compiled,  # type: ignore[arg-type]
                None,
                None,
                compiled.diagnostic or "CUDA Replay compilation failed",
                generation.attempts,
            )
        try:
            replay_fingerprint = self._fingerprint(artifact, compiled, case)
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                compiled,  # type: ignore[arg-type]
                None,
                None,
                f"Replay observation failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        comparison = self.comparator.compare(source_fingerprint, replay_fingerprint)
        return Gate1ArmEvidence(
            mode,
            generation,
            compiled,  # type: ignore[arg-type]
            replay_fingerprint,  # type: ignore[arg-type]
            comparison,  # type: ignore[arg-type]
            physical_calls=generation.attempts,
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: CudaKernelSource,
        source_case: CudaCompileCase,
    ) -> Gate1ExecutionResult:
        digest = canonical_artifact_digest(artifact, observer_level=self.config.observer_level)
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                empty,
                None,
                0,
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
            )
        try:
            _validate_schedule(artifact, source_case.schedule)
            source_fingerprint = self._fingerprint(artifact, source_compile, source_case)
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                normal,
                None,
                0,
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
            )
        normal_candidates = _normal_arms(
            self, artifact, source_fingerprint, source_case
        )
        selected_normal_index = _selected_normal_index(normal_candidates)
        normal = normal_candidates[selected_normal_index]
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                source_fingerprint,  # type: ignore[arg-type]
                normal,
                None,
                len(normal_candidates),
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
                normal_candidates=normal_candidates,
                selected_normal_index=selected_normal_index,
            )
        masked_evidence = None
        masked_calls = 0
        if self.config.run_masked_j:
            assert self.masked_writer is not None
            masked_calls = 1
            masked_evidence = self._arm(
                self.masked_writer,
                artifact,
                source_fingerprint,
                source_case,
                masked=True,
            )
        return Gate1ExecutionResult(
            Gate1Status.REPLAY_CLOSED,
            digest,
            source_compile,  # type: ignore[arg-type]
            source_fingerprint,  # type: ignore[arg-type]
            normal,
            masked_evidence,
            len(normal_candidates),
            masked_calls,
            observer_profile=CUDA_OBSERVER_PROFILE,
            normal_candidates=normal_candidates,
            selected_normal_index=selected_normal_index,
            masked_j_independent_writer=(
                True if self.config.run_masked_j else None
            ),
        )


class TileLangGate1Runner:
    def __init__(
        self,
        *,
        compiler: TileLangCompiler,
        writer: ReplayWriter | tuple[ReplayWriter, ...],
        masked_writer: ReplayWriter | None = None,
        comparator: TileLangComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.config = config or Gate1Config()
        self.writers = _replay_writers(writer)
        if self.config.run_masked_j:
            if masked_writer is None:
                raise ValueError("masked-J requires a fresh independent Replay writer")
            if id(masked_writer) in {id(item) for item in self.writers}:
                raise ValueError("masked-J writer must differ from every normal writer")
        elif masked_writer is not None:
            raise ValueError("masked-J writer was provided while the audit is disabled")
        self.writer = self.writers[0]
        self.masked_writer = masked_writer
        self.comparator = comparator or TileLangComparator()

    def _fingerprint(
        self,
        artifact: Gate1Artifact,
        result: TileLangCompileResult,
        case: TileLangCompileCase,
    ) -> TileLangFingerprint:
        if result.tir is None or result.ptx is None:
            raise ValueError("TileLang compile result lacks TIR/PTX")
        interface = _artifact_interface(artifact, case.parameters)
        return fingerprint_tilelang(
            result.tir,
            result.ptx,
            interface=interface,
            target={"name": case.target},
            schedule=case.schedule,
        )

    def _arm(
        self,
        writer: ReplayWriter,
        artifact: Gate1Artifact,
        source_fingerprint: TileLangFingerprint,
        case: TileLangCompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = writer.generate(artifact, masked=masked)
        except Exception as exc:
            attempt = getattr(exc, "attempt", None)
            return Gate1ArmEvidence(
                mode, None, None, None, None,
                f"generation failed: {type(exc).__name__}: {exc}",
                getattr(attempt, "attempts", 0),
            )
        try:
            replay_source = TileLangKernelSource.from_replay(
                generation.code, generation.function_name
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, None, None, None,
                f"Replay policy failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        compiled = self.compiler.compile(replay_source, case)
        if not compiled.ok:
            return Gate1ArmEvidence(
                mode, generation, compiled, None, None,  # type: ignore[arg-type]
                compiled.diagnostic or "TileLang Replay compilation failed",
                generation.attempts,
            )
        try:
            replay_fingerprint = self._fingerprint(artifact, compiled, case)
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, compiled, None, None,  # type: ignore[arg-type]
                f"Replay observation failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        comparison = self.comparator.compare(source_fingerprint, replay_fingerprint)
        return Gate1ArmEvidence(
            mode,
            generation,
            compiled,  # type: ignore[arg-type]
            replay_fingerprint,  # type: ignore[arg-type]
            comparison,  # type: ignore[arg-type]
            physical_calls=generation.attempts,
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: TileLangKernelSource,
        source_case: TileLangCompileCase,
    ) -> Gate1ExecutionResult:
        digest = canonical_artifact_digest(artifact, observer_level=self.config.observer_level)
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                empty,
                None,
                0,
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
            )
        try:
            _validate_schedule(artifact, source_case.schedule)
            source_fingerprint = self._fingerprint(artifact, source_compile, source_case)
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                normal,
                None,
                0,
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
            )
        normal_candidates = _normal_arms(
            self, artifact, source_fingerprint, source_case
        )
        selected_normal_index = _selected_normal_index(normal_candidates)
        normal = normal_candidates[selected_normal_index]
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                source_fingerprint,  # type: ignore[arg-type]
                normal,
                None,
                len(normal_candidates),
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
                normal_candidates=normal_candidates,
                selected_normal_index=selected_normal_index,
            )
        masked_evidence = None
        masked_calls = 0
        if self.config.run_masked_j:
            assert self.masked_writer is not None
            masked_calls = 1
            masked_evidence = self._arm(
                self.masked_writer,
                artifact,
                source_fingerprint,
                source_case,
                masked=True,
            )
        return Gate1ExecutionResult(
            Gate1Status.REPLAY_CLOSED,
            digest,
            source_compile,  # type: ignore[arg-type]
            source_fingerprint,  # type: ignore[arg-type]
            normal,
            masked_evidence,
            len(normal_candidates),
            masked_calls,
            observer_profile=TILELANG_OBSERVER_PROFILE,
            normal_candidates=normal_candidates,
            selected_normal_index=selected_normal_index,
            masked_j_independent_writer=(
                True if self.config.run_masked_j else None
            ),
        )


__all__ = ["CudaGate1Runner", "TileLangGate1Runner"]
