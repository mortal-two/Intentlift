from __future__ import annotations

import json

import pytest

from intentir.extraction.providers import client
from intentir.extraction.providers.client import (
    _resolve_responses_request_policy,
    _responses_payload,
    call_responses,
)
from intentir.extraction.providers.provider import ProviderConfig


def _provider(*, effort: str = "high") -> ProviderConfig:
    return ProviderConfig(
        name="test",
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="gpt-test",
        reasoning_effort=effort,
    )


def test_responses_payload_remains_compatible_without_text_format() -> None:
    payload = _responses_payload("system", "user", _provider(), None, None)

    assert payload == {
        "model": "gpt-test",
        "instructions": "system",
        "input": "user",
        "reasoning": {"effort": "high"},
    }


def test_responses_payload_carries_json_schema_text_format() -> None:
    text_format = {
        "type": "json_schema",
        "name": "artifact",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "string"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }

    payload = _responses_payload(
        "system", "user", _provider(), "xhigh", text_format
    )

    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["text"] == {"format": text_format}


def test_xhigh_request_policy_uses_one_long_nonduplicated_request() -> None:
    assert _resolve_responses_request_policy(
        _provider(), "xhigh", None, None
    ) == (1, 3600.0)
    assert _resolve_responses_request_policy(
        _provider(effort="xhigh"), None, None, None
    ) == (1, 3600.0)


def test_non_xhigh_request_policy_preserves_historical_defaults() -> None:
    assert _resolve_responses_request_policy(
        _provider(), None, None, None
    ) == (4, 600.0)
    assert _resolve_responses_request_policy(
        _provider(effort="xhigh"), "high", None, None
    ) == (4, 600.0)


def test_explicit_request_policy_overrides_xhigh_defaults() -> None:
    assert _resolve_responses_request_policy(
        _provider(effort="xhigh"), None, 7, 42.5
    ) == (7, 42.5)


@pytest.mark.parametrize(
    ("max_retries", "timeout", "message"),
    [(0, None, "max_retries"), (None, 0.0, "timeout")],
)
def test_request_policy_rejects_non_positive_values(
    max_retries: int | None, timeout: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_responses_request_policy(
            _provider(), None, max_retries, timeout
        )


def test_call_responses_passes_resolved_xhigh_timeout_to_urlopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"output_text": "ok"}).encode()

    def _fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        observed["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(client.urllib.request, "urlopen", _fake_urlopen)

    reply = call_responses(
        "system", "user", provider=_provider(effort="xhigh")
    )

    assert reply.text == "ok"
    assert observed == {"timeout": 3600.0}
