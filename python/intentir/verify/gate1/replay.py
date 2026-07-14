"""Replay-agent adapters for SCRC Gate1.

The replay writer is intentionally handed only ``Gate1Artifact.rewriter_view``
through :mod:`.prompt`; source text, location maps and concrete C values never
enter this API.  The deterministic compiler/comparator remain independent of
which model (or static fixture) proposes the replay code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from intentir.extraction.providers.client import ProviderError, call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import extract_python_module
from .diagnose import RepairFeedback
from .model import Gate1Artifact
from .prompt import build_rewriter_prompt
from .rules import RuleRegistry


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    prompt_sha256: str
    requested_model: str
    generation_id: str
    code_sha256: str
    response_model: str | None = None
    response_id: str | None = None
    masked: bool = False
    feedback: RepairFeedback | None = None


class ReplayWriter(Protocol):
    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        feedback: RepairFeedback | None = None,
        base_candidate: ReplayGeneration | None = None,
        masked: bool = False,
    ) -> ReplayGeneration: ...


_SYSTEM = """You are the source-blind replay writer in IntentIR SCRC Gate1.
Use only the supplied replay packet and selected rule cards. Never request,
infer, or reproduce source text. Return exactly one importable Triton Python
module with no Markdown fence or prose. Compiler/comparator feedback may be
used only to repair the requested implementation; J remains the semantic
authority and concrete schedule values remain unavailable."""


def _augment_prompt(
    prompt: str,
    *,
    feedback: RepairFeedback | None,
    base_candidate: ReplayGeneration | None,
) -> str:
    if feedback is None:
        return prompt
    if base_candidate is None:
        raise ValueError("repair feedback requires a base replay candidate")
    return (
        prompt
        + "\n\nCURRENT_REPLAY_CANDIDATE (model-generated, not source):\n"
        + base_candidate.code
        + "\n\nALLOWLISTED_REPAIR_FEEDBACK:\n"
        + json.dumps(feedback.to_json(), ensure_ascii=False, sort_keys=True)
        + "\nRepair only the allowed scope and re-emit the complete module."
    )


class ResponsesReplayWriter:
    """OpenAI-compatible Responses transport used by the existing project."""

    def __init__(
        self,
        *,
        provider: ProviderConfig | None = None,
        model_candidates: Sequence[str] | None = None,
        effort: str = "high",
        max_retries: int = 2,
        timeout: float = 600.0,
    ) -> None:
        self.provider = provider or get_provider()
        self.model_candidates = tuple(model_candidates or (self.provider.model,))
        if not self.model_candidates:
            raise ValueError("model_candidates must not be empty")
        self.effort = effort
        self.max_retries = max_retries
        self.timeout = timeout
        self._fallback_generation = 0

    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        feedback: RepairFeedback | None = None,
        base_candidate: ReplayGeneration | None = None,
        masked: bool = False,
    ) -> ReplayGeneration:
        user_input = _augment_prompt(
            build_rewriter_prompt(artifact, registry, masked=masked),
            feedback=feedback,
            base_candidate=base_candidate,
        )
        prompt_sha = hashlib.sha256(
            (_SYSTEM + "\x00" + user_input).encode("utf-8")
        ).hexdigest()
        last_error: Exception | None = None
        for model in self.model_candidates:
            provider = replace(self.provider, model=model)
            try:
                reply = call_responses(
                    _SYSTEM,
                    user_input,
                    provider=provider,
                    effort=self.effort,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
                function_name = getattr(
                    artifact.interface, "function_name", artifact.interface.kernel_id
                )
                code = extract_python_module(
                    reply.text, required_function=function_name
                )
                code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
                response_id = reply.raw.get("id")
                if response_id:
                    generation_id = f"responses:{response_id}"
                else:
                    self._fallback_generation += 1
                    generation_id = (
                        f"responses-fallback:{self._fallback_generation}:{code_sha}"
                    )
                return ReplayGeneration(
                    code=code,
                    prompt_sha256=prompt_sha,
                    requested_model=model,
                    generation_id=generation_id,
                    code_sha256=code_sha,
                    response_model=reply.raw.get("model"),
                    response_id=response_id,
                    masked=masked,
                    feedback=feedback,
                )
            except ProviderError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise last_error


class StaticReplayWriter:
    """Deterministic replay source for tests, cached artifacts and evaluation."""

    def __init__(
        self,
        normal: str | Sequence[str],
        *,
        masked: str | Sequence[str] | None = None,
    ) -> None:
        self._normal = list(normal) if not isinstance(normal, str) else [normal]
        masked_values: str | Sequence[str] = () if masked is None else masked
        self._masked = (
            list(masked_values)
            if not isinstance(masked_values, str)
            else [masked_values]
        )
        self._normal_index = 0
        self._masked_index = 0

    @staticmethod
    def _take(values: list[str], index: int) -> tuple[str, int]:
        if not values:
            raise RuntimeError("StaticReplayWriter has no replay source")
        return values[min(index, len(values) - 1)], index + 1

    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        feedback: RepairFeedback | None = None,
        base_candidate: ReplayGeneration | None = None,
        masked: bool = False,
    ) -> ReplayGeneration:
        # Build the prompt even for static fixtures so isolation/registry errors
        # are exercised exactly as they are for a live writer.
        prompt = _augment_prompt(
            build_rewriter_prompt(artifact, registry, masked=masked),
            feedback=feedback,
            base_candidate=base_candidate,
        )
        if masked:
            invocation = self._masked_index
            code, self._masked_index = self._take(
                self._masked, self._masked_index
            )
        else:
            invocation = self._normal_index
            code, self._normal_index = self._take(
                self._normal, self._normal_index
            )
        function_name = getattr(
            artifact.interface, "function_name", artifact.interface.kernel_id
        )
        code = extract_python_module(code, required_function=function_name)
        code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        mode = "masked" if masked else "normal"
        return ReplayGeneration(
            code=code,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            requested_model="static",
            generation_id=f"static:{mode}:{invocation}:{code_sha}",
            code_sha256=code_sha,
            response_model="static",
            masked=masked,
            feedback=feedback,
        )


__all__ = [
    "ReplayGeneration",
    "ReplayWriter",
    "ResponsesReplayWriter",
    "StaticReplayWriter",
]
