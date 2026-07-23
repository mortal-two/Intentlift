from __future__ import annotations

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.extraction.providers.client import LLMReply
from intentir.verify.gate1 import replay as replay_module
from intentir.verify.gate1.replay import (
    ReplayAttempt,
    ReplayGenerationError,
    ResponsesReplayWriter,
)


def test_rejected_replay_response_is_retained_with_physical_call_count(
    monkeypatch,
) -> None:
    response = """import triton
from triton.language.extra.libdevice import tanh
@triton.jit
def scrc_k0(p0):
    return tanh(p0)
"""
    monkeypatch.setattr(
        replay_module,
        "call_responses",
        lambda *_args, **_kwargs: LLMReply(
            response,
            "test-model",
            {"id": "response-1", "model": "test-model"},
            attempts=2,
            elapsed_ms=12.0,
        ),
    )
    writer = ResponsesReplayWriter()

    with pytest.raises(ReplayGenerationError) as captured:
        writer.generate(make_reduce_artifact())

    assert captured.value.attempt.attempts == 2
    assert len(writer.records) == 1
    assert isinstance(writer.records[0], ReplayAttempt)
    assert writer.records[0].response_text == response
