from __future__ import annotations

from intentir.extraction.providers.client import _responses_payload
from intentir.extraction.providers.provider import ProviderConfig


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="test",
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="gpt-test",
        reasoning_effort="high",
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
