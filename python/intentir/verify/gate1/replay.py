"""One-shot source-blind Replay writers for direct ``J + C + pi`` packets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig

from .codegen import extract_python_module
from .model import Gate1Artifact
from .prompt import build_rewriter_prompt, replay_packet_digest


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    function_name: str
    model: str
    response_id: str | None
    packet_digest: str
    prompt_digest: str
    context_receipt: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)


class ReplayWriter(Protocol):
    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration: ...


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    value = raw.get("usage")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _context_receipt(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    trace: list[dict[str, str]] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and "search" in item_type:
                trace.append(
                    {
                        "type": item_type,
                        "id": item.get("id") if isinstance(item.get("id"), str) else "",
                    }
                )
    return {
        "context_sha256": hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "retrieval_trace": trace,
    }


@dataclass
class ResponsesReplayWriter:
    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend_context: Mapping[str, Any] | None = None

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(
            artifact,
            masked=masked,
            frontend_context=self.frontend_context,
        )
        packet = json.loads(user)
        function_name = packet["harness"]["function_name"]
        reply = call_responses(
            instructions,
            user,
            provider=self.provider,
            effort=self.effort,
        )
        code = extract_python_module(
            reply.text,
            required_function=function_name,
        )
        prompt_digest = hashlib.sha256(
            (instructions + "\0" + user).encode()
        ).hexdigest()
        return ReplayGeneration(
            code=code,
            function_name=function_name,
            model=reply.model,
            response_id=(
                reply.raw.get("id")
                if isinstance(reply.raw.get("id"), str) else None
            ),
            packet_digest=replay_packet_digest(packet),
            prompt_digest=prompt_digest,
            context_receipt=_context_receipt(
                reply.raw, packet.get("frontend_context", {})
            ),
            usage=_usage(reply.raw),
        )


@dataclass
class StaticReplayWriter:
    code: str
    model: str = "static-test-writer"
    calls: list[bool] = field(default_factory=list, init=False)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(artifact, masked=masked)
        packet = json.loads(user)
        function_name = packet["harness"]["function_name"]
        code = extract_python_module(self.code, required_function=function_name)
        self.calls.append(masked)
        return ReplayGeneration(
            code=code,
            function_name=function_name,
            model=self.model,
            response_id=None,
            packet_digest=replay_packet_digest(packet),
            prompt_digest=hashlib.sha256(
                (instructions + "\0" + user).encode()
            ).hexdigest(),
            context_receipt={
                "context_sha256": hashlib.sha256(
                    json.dumps(
                        packet.get("frontend_context", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "retrieval_trace": [],
            },
        )


__all__ = [
    "ReplayGeneration", "ReplayWriter", "ResponsesReplayWriter",
    "StaticReplayWriter",
]
