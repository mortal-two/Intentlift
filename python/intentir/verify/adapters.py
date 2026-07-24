"""Thin adapters from the unified SCRC lifecycle to active frontend runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .gate1.compiler import KernelSource
from .gate1.cuda_compiler import CudaKernelSource
from .gate1.model import Gate1Artifact
from .gate1.orchestrate import Gate1ExecutionResult
from .gate1.tilelang_compiler import TileLangKernelSource
from .gate2.model import Gate2SuiteResult
from .gate2.orchestrate import Gate2Runner, Gate2Suite
from .scrc_runner import HarnessBindingInstance, SCRCRunRequest


@dataclass
class BoundGate1Adapter:
    """Bind one fresh low-level Gate1 runner to the request's actual c0 case."""

    runner: Any

    def run(
        self,
        artifact: Gate1Artifact,
        request: SCRCRunRequest,
    ) -> Gate1ExecutionResult:
        return self.runner.run(
            artifact,
            source=request.source.gate_source,
            source_case=request.binding.gate1_case,
        )


Gate2SuiteFactory = Callable[
    [Gate1Artifact, Gate1ExecutionResult, HarnessBindingInstance],
    Gate2Suite,
]


@dataclass
class TritonGate2ExecutionAdapter:
    """Create hidden cases only after Gate1 and execute the exact accepted R."""

    runner: Gate2Runner
    suite_factory: Gate2SuiteFactory

    def execute(
        self,
        artifact: Gate1Artifact,
        gate1: Gate1ExecutionResult,
        request: SCRCRunRequest,
    ) -> Gate2SuiteResult:
        if request.frontend != "triton":
            raise ValueError("Triton Gate2 adapter received a non-Triton request")
        if not isinstance(request.source.gate_source, KernelSource):
            raise TypeError("Triton Gate2 requires the trusted source KernelSource")
        generation = gate1.normal.generation
        if generation is None:
            raise ValueError("Gate2 requires the exact Gate1-accepted Replay")
        replay = KernelSource.from_replay(
            generation.code, generation.function_name
        )
        # The factory is invoked here, never before Gate1 closure, and receives
        # the one Harness binding instance used by Gate1.
        suite = self.suite_factory(artifact, gate1, request.binding)
        if not isinstance(suite, Gate2Suite):
            raise TypeError("Gate2 suite_factory must return Gate2Suite")
        if suite.c_values != request.binding.c_values:
            raise ValueError("Gate2 suite must use the same c0 as Gate1")
        if any(case.j is not artifact.j for case in suite.cases):
            raise ValueError("every Gate2 RefEval case must use the accepted J")
        return self.runner.run_suite(request.source.gate_source, replay, suite)


@dataclass
class NativeGate2ExecutionAdapter:
    """Run CUDA or TileLang Source/Replay with the shared three-way Gate2 judge."""

    frontend: str
    runner: Gate2Runner
    suite_factory: Gate2SuiteFactory

    def __post_init__(self) -> None:
        if self.frontend not in {"cuda", "tilelang"}:
            raise ValueError("native Gate2 frontend must be cuda or tilelang")

    def execute(
        self,
        artifact: Gate1Artifact,
        gate1: Gate1ExecutionResult,
        request: SCRCRunRequest,
    ) -> Gate2SuiteResult:
        if request.frontend != self.frontend:
            raise ValueError(
                f"{self.frontend} Gate2 adapter received {request.frontend!r}"
            )
        source = request.source.gate_source
        generation = gate1.normal.generation
        if generation is None:
            raise ValueError("Gate2 requires the exact Gate1-accepted Replay")
        if self.frontend == "cuda":
            if not isinstance(source, CudaKernelSource):
                raise TypeError("CUDA Gate2 requires CudaKernelSource")
            replay = CudaKernelSource.from_replay(
                generation.code, generation.function_name
            )
        else:
            if not isinstance(source, TileLangKernelSource):
                raise TypeError("TileLang Gate2 requires TileLangKernelSource")
            replay = TileLangKernelSource.from_replay(
                generation.code, generation.function_name
            )
        suite = self.suite_factory(artifact, gate1, request.binding)
        if not isinstance(suite, Gate2Suite):
            raise TypeError("native Gate2 suite_factory must return Gate2Suite")
        if suite.c_values != request.binding.c_values:
            raise ValueError("native Gate2 suite must use the same c0 as Gate1")
        if any(case.j is not artifact.j for case in suite.cases):
            raise ValueError("every native Gate2 case must use the accepted J")
        return self.runner.run_suite(source, replay, suite)  # type: ignore[arg-type]


__all__ = [
    "BoundGate1Adapter",
    "Gate2SuiteFactory",
    "NativeGate2ExecutionAdapter",
    "TritonGate2ExecutionAdapter",
]
