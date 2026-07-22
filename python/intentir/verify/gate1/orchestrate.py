"""Gate1 realization closure with an independent optional masked-J audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .compare import TTIRComparator, TTIRComparison
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .model import (
    Gate1Artifact,
    Gate1Status,
    JNecessityStatus,
    ParameterKind,
    ScheduleSlotKind,
    opaque_replay_abi,
)
from .replay import ReplayGeneration, ReplayWriter
from .ttir import TTIRFingerprint, fingerprint_ttir


GATE1_PROTOCOL = "llm_direct_pi_gate1_level2_v1"


@dataclass(frozen=True)
class Gate1Config:
    observer_level: int = 2
    run_masked_j: bool = True

    def __post_init__(self) -> None:
        if self.observer_level != 2:
            raise ValueError("the active strong Gate1 paths require observer level 2")


@dataclass(frozen=True)
class Gate1ArmEvidence:
    mode: str
    generation: ReplayGeneration | None
    compile_result: CompileResult | None
    fingerprint: TTIRFingerprint | None
    comparison: TTIRComparison | None
    diagnostic: str | None = None

    @property
    def hard_closed(self) -> bool:
        return self.comparison is not None and self.comparison.hard_pass


@dataclass(frozen=True)
class Gate1ExecutionResult:
    status: Gate1Status
    artifact_digest: str
    source_compile: CompileResult
    source_fingerprint: TTIRFingerprint | None
    normal: Gate1ArmEvidence
    masked_j: Gate1ArmEvidence | None
    normal_calls: int
    masked_j_calls: int
    protocol: str = GATE1_PROTOCOL

    @property
    def passed(self) -> bool:
        return self.status is Gate1Status.REPLAY_CLOSED

    @property
    def j_necessity_status(self) -> JNecessityStatus:
        """Report masked-J separately from the main Gate1 status."""

        if self.masked_j is None or self.masked_j.comparison is None:
            return JNecessityStatus.INCONCLUSIVE
        if self.masked_j.hard_closed:
            return JNecessityStatus.NOT_PROVED
        return JNecessityStatus.PROVED

    @property
    def replay_code(self) -> str | None:
        return self.normal.generation.code if self.normal.generation is not None else None

    def to_json(self) -> dict[str, Any]:
        def arm(value: Gate1ArmEvidence | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "mode": value.mode,
                "generation": (
                    {
                        "model": value.generation.model,
                        "function_name": value.generation.function_name,
                        "entry_function_name": value.generation.entry_function_name,
                        "response_model": value.generation.response_model,
                        "response_id": value.generation.response_id,
                        "packet_digest": value.generation.packet_digest,
                        "prompt_digest": value.generation.prompt_digest,
                        "request_sha256": value.generation.request_sha256,
                        "response_sha256": value.generation.response_sha256,
                        "raw_response_sha256": value.generation.raw_response_sha256,
                        "attempts": value.generation.attempts,
                        "elapsed_ms": value.generation.elapsed_ms,
                        "context_receipt": dict(value.generation.context_receipt),
                        "usage": dict(value.generation.usage),
                    }
                    if value.generation is not None else None
                ),
                "compile": (
                    value.compile_result.to_payload()
                    if value.compile_result is not None else None
                ),
                "fingerprint": (
                    value.fingerprint.to_dict()
                    if value.fingerprint is not None else None
                ),
                "comparison": (
                    value.comparison.to_dict()
                    if value.comparison is not None else None
                ),
                "diagnostic": value.diagnostic,
            }

        return {
            "protocol": self.protocol,
            "status": self.status.value,
            "j_necessity_status": self.j_necessity_status.value,
            "artifact_digest": self.artifact_digest,
            "source_compile": self.source_compile.to_payload(),
            "source_fingerprint": (
                self.source_fingerprint.to_dict()
                if self.source_fingerprint is not None else None
            ),
            "normal": arm(self.normal),
            "masked_j": arm(self.masked_j),
            "llm_calls": {
                "normal": self.normal_calls,
                "masked_j": self.masked_j_calls,
            },
        }


def canonical_artifact_digest(
    artifact: Gate1Artifact, *, observer_level: int = 2
) -> str:
    """Compute an audit identifier; it is not a verdict or protocol gate."""

    payload = {
        "schema": artifact.SCHEMA,
        "j": artifact.j.to_json(),
        "c": artifact.c.to_json(),
        "pi": artifact.pi.to_json(),
        "observer_level": observer_level,
        "protocol": GATE1_PROTOCOL,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _opaque_python_interface(
    artifact: Gate1Artifact,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Alpha-normalize Python signature spelling by declaration position."""

    if not metadata:
        return {}
    result = dict(metadata)
    raw_names = result.get("arg_names")
    if not isinstance(raw_names, (list, tuple)) or not all(
        isinstance(name, str) for name in raw_names
    ):
        raise ValueError("compiler metadata lacks an ordered Python signature")
    signature_parameters = artifact.j.signature.parameters
    c_constexprs = [
        slot for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.CONSTEXPR
    ]
    if len(raw_names) != len(signature_parameters) + len(c_constexprs):
        raise ValueError("compiled Python signature arity differs from artifact harness")
    constexpr_indices = result.get("constexpr_indices", [])
    if not isinstance(constexpr_indices, (list, tuple)) or not all(
        isinstance(index, int) and not isinstance(index, bool)
        for index in constexpr_indices
    ):
        raise ValueError("compiler constexpr indices are malformed")
    signature_constexpr = [
        index
        for index, parameter in enumerate(signature_parameters)
        if parameter.kind is ParameterKind.CONSTEXPR
    ]
    expected_constexpr = [
        *signature_constexpr,
        *range(len(signature_parameters), len(signature_parameters) + len(c_constexprs)),
    ]
    if sorted(constexpr_indices) != expected_constexpr:
        raise ValueError("compiled constexpr positions differ from artifact harness")

    opaque_function, parameter_names, schedule_names = opaque_replay_abi(artifact)
    aliases = [
        parameter_names[parameter.name]
        for parameter in signature_parameters
    ] + [
        schedule_names[slot.name] for slot in c_constexprs
    ]
    result["arg_names"] = aliases
    result["constexpr_indices"] = expected_constexpr
    result["constexprs"] = [aliases[index] for index in expected_constexpr]
    if "function_name" in result:
        result["function_name"] = opaque_function
    return result


def _replay_case(
    artifact: Gate1Artifact,
    source_case: CompileCase,
    source_compile: CompileResult,
) -> CompileCase:
    _function, parameter_names, schedule_names = opaque_replay_abi(artifact)
    signature_parameters = artifact.j.signature.parameters
    runtime_parameters = [
        item for item in signature_parameters
        if item.kind is not ParameterKind.CONSTEXPR
    ]
    semantic_constexprs = [
        item for item in signature_parameters
        if item.kind is ParameterKind.CONSTEXPR
    ]
    c_constexprs = [
        slot for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.CONSTEXPR
    ]
    metadata_names = source_compile.python_interface.get("arg_names")
    constexpr_indices = source_compile.python_interface.get("constexpr_indices")
    if metadata_names:
        if not isinstance(metadata_names, (list, tuple)) or not all(
            isinstance(name, str) for name in metadata_names
        ):
            raise ValueError("source compiler returned malformed argument names")
        if len(metadata_names) != len(signature_parameters) + len(c_constexprs):
            raise ValueError("source signature arity differs from artifact harness")
        if not isinstance(constexpr_indices, (list, tuple)):
            raise ValueError("source compiler returned malformed constexpr positions")
        constexpr_positions = {int(index) for index in constexpr_indices}
        source_runtime_names = [
            name for index, name in enumerate(metadata_names)
            if index not in constexpr_positions
        ]
        source_constexpr_names = [
            name for index, name in enumerate(metadata_names)
            if index in constexpr_positions
        ]
        if set(source_runtime_names) != set(source_case.arguments):
            raise ValueError("source runtime bindings differ from compiled signature")
        argument_values = [source_case.arguments[name] for name in source_runtime_names]
    else:
        argument_values = list(source_case.arguments.values())
        source_constexpr_names = list(source_case.schedule)[:
            len(semantic_constexprs) + len(c_constexprs)
        ]
    if len(runtime_parameters) != len(argument_values):
        raise ValueError(
            "source invocation arity differs from artifact positional harness"
        )
    dtype_aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
        "i1": "bool",
    }

    def dtype_key(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.removeprefix("torch.")
        return dtype_aliases.get(stripped, stripped)

    for parameter, value in zip(runtime_parameters, argument_values, strict=True):
        if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            if value.kind != "tensor":
                raise ValueError(
                    f"source position for {parameter.name!r} must be a tensor"
                )
            if parameter.rank is not None and len(value.shape or ()) != parameter.rank:
                raise ValueError(
                    f"source rank for {parameter.name!r} differs from artifact harness"
                )
        elif parameter.kind is ParameterKind.SCALAR and value.kind != "scalar":
            raise ValueError(
                f"source position for {parameter.name!r} must be a scalar"
            )
        if (
            parameter.dtype is not None
            and value.dtype is not None
            and dtype_key(parameter.dtype) != dtype_key(value.dtype)
        ):
            raise ValueError(
                f"source dtype for {parameter.name!r} differs from artifact harness"
            )
    arguments = {
        parameter_names[parameter.name]: value
        for parameter, value in zip(runtime_parameters, argument_values, strict=True)
    }
    slots = {slot.name: slot for slot in artifact.c.slots}
    if len(source_constexpr_names) != len(semantic_constexprs) + len(c_constexprs):
        raise ValueError("source constexpr bindings differ from artifact harness")
    schedule: dict[str, Any] = {}
    consumed_source_schedule: set[str] = set()
    for parameter, source_name in zip(
        semantic_constexprs,
        source_constexpr_names[: len(semantic_constexprs)],
        strict=True,
    ):
        if source_name not in source_case.schedule:
            raise ValueError(f"source constexpr {source_name!r} is unbound")
        # Semantic compile-time values are Harness bindings.  Their logical
        # consequence is already in J; no concrete value enters C or Replay.
        expected = source_case.schedule[source_name]
        schedule[parameter_names[parameter.name]] = expected
        consumed_source_schedule.add(source_name)

    c_source_names = source_constexpr_names[len(semantic_constexprs):]
    for slot, source_name in zip(c_constexprs, c_source_names, strict=True):
        if source_name not in source_case.schedule:
            raise ValueError(f"source C constexpr {source_name!r} is unbound")
        expected = artifact.c.values.get(slot.name)
        if source_case.schedule[source_name] != expected:
            raise ValueError(
                f"source constexpr position for {slot.name!r} differs from C"
            )
        schedule[schedule_names[slot.name]] = expected
        consumed_source_schedule.add(source_name)

    source_compiler_names = [
        name for name in source_case.schedule if name not in consumed_source_schedule
    ]
    compiler_slot_names = [
        slot.name for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.COMPILER
    ]
    if len(source_compiler_names) != len(compiler_slot_names):
        raise ValueError("source compiler-option bindings differ from C")
    compiler_roles: set[str] = set()
    for source_name, slot_name in zip(
        source_compiler_names, compiler_slot_names, strict=True
    ):
        slot = slots[slot_name]
        if slot.kind is not ScheduleSlotKind.COMPILER:
            raise ValueError("artifact compiler slot has the wrong C kind")
        if slot.role in compiler_roles:
            raise ValueError("active compiler C roles must be unique")
        compiler_roles.add(slot.role)
        expected = artifact.c.values.get(slot_name)
        if source_case.schedule[source_name] != expected:
            raise ValueError(
                f"source compiler-option position for {slot_name!r} differs from C"
            )
        schedule[slot.role] = expected
    return CompileCase(
        arguments=arguments,
        grid=source_case.grid,
        schedule=schedule,
        device=source_case.device,
    )


class Gate1Runner:
    def __init__(
        self,
        *,
        compiler: TritonCompiler,
        writer: ReplayWriter,
        comparator: TTIRComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.writer = writer
        self.comparator = comparator or TTIRComparator()
        self.config = config or Gate1Config()

    def _arm(
        self,
        artifact: Gate1Artifact,
        source_fingerprint: TTIRFingerprint,
        replay_case: CompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = self.writer.generate(artifact, masked=masked)
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, None, None, None, None,
                f"generation failed: {type(exc).__name__}: {exc}",
            )
        replay_source = KernelSource.from_replay(
            generation.code, generation.function_name
        )
        compile_result = self.compiler.compile(replay_source, replay_case)
        if not compile_result.ok or compile_result.ttir is None:
            return Gate1ArmEvidence(
                mode, generation, compile_result, None, None,
                compile_result.diagnostic or "replay compilation failed",
            )
        try:
            replay_fingerprint = fingerprint_ttir(
                compile_result.ttir,
                python_interface=_opaque_python_interface(
                    artifact, compile_result.python_interface
                ),
                schedule=artifact.c.values,
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, compile_result, None, None,
                f"replay observation failed: {type(exc).__name__}: {exc}",
            )
        comparison = self.comparator.compare(
            source_fingerprint, replay_fingerprint
        )
        return Gate1ArmEvidence(
            mode, generation, compile_result, replay_fingerprint, comparison
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: KernelSource,
        source_case: CompileCase,
    ) -> Gate1ExecutionResult:
        digest = canonical_artifact_digest(
            artifact, observer_level=self.config.observer_level
        )
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok or source_compile.ttir is None:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,
                None,
                empty,
                None,
                0,
                0,
            )
        try:
            replay_case = _replay_case(artifact, source_case, source_compile)
            source_fingerprint = fingerprint_ttir(
                source_compile.ttir,
                python_interface=_opaque_python_interface(
                    artifact, source_compile.python_interface
                ),
                schedule=artifact.c.values,
            )
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                None,
                normal,
                None,
                0,
                0,
            )

        normal = self._arm(
            artifact, source_fingerprint, replay_case, masked=False
        )
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                source_fingerprint,
                normal,
                None,
                1,
                0,
            )

        masked_evidence: Gate1ArmEvidence | None = None
        masked_calls = 0
        status = Gate1Status.REPLAY_CLOSED
        if self.config.run_masked_j:
            masked_calls = 1
            masked_evidence = self._arm(
                artifact, source_fingerprint, replay_case, masked=True
            )
        return Gate1ExecutionResult(
            status,
            digest,
            source_compile,
            source_fingerprint,
            normal,
            masked_evidence,
            1,
            masked_calls,
        )


__all__ = [
    "GATE1_PROTOCOL", "Gate1ArmEvidence", "Gate1Config",
    "Gate1ExecutionResult", "Gate1Runner", "canonical_artifact_digest",
]
