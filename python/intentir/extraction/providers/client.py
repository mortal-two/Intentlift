"""Minimal HTTP client for the LLM provider (OpenAI ``responses`` wire API).

Dependency-free (stdlib ``urllib``) so extraction runs without extra packages.
The API key is read from :mod:`provider` and is NEVER logged, printed, or put in
an exception message — only ``Authorization`` header bytes that urllib sends.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .provider import ProviderConfig, get_provider


DEFAULT_RESPONSES_MAX_RETRIES = 4
DEFAULT_RESPONSES_TIMEOUT_SECONDS = 600.0
XHIGH_RESPONSES_MAX_RETRIES = 2
XHIGH_RESPONSES_TIMEOUT_SECONDS = 1800.0


class ProviderError(RuntimeError):
    """Raised when the provider call fails after retries. Never carries the key."""


@dataclass
class LLMReply:
    text: str
    model: str
    raw: dict


def _extract_output_text(data: dict) -> str:
    """Pull the concatenated assistant text out of a responses-API payload."""
    # Preferred: the SDK convenience field.
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        if item.get("type") not in (None, "message"):
            continue
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text") and "text" in c:
                chunks.append(c["text"])
    return "".join(chunks)


def _responses_payload(
    instructions: str,
    user_input: str,
    provider: ProviderConfig,
    effort: str | None,
    text_format: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": effort or provider.reasoning_effort},
    }
    if text_format is not None:
        payload["text"] = {"format": dict(text_format)}
    return payload


def _resolve_responses_request_policy(
    provider: ProviderConfig,
    effort: str | None,
    max_retries: int | None,
    timeout: float | None,
) -> tuple[int, float]:
    """Resolve request limits without duplicating long-running xhigh calls.

    ``xhigh`` responses can legitimately take longer than the ordinary request
    window.  Give them one long window plus one retry, while preserving the
    historical defaults for every other effort.  Explicit caller values always
    win.
    """
    effective_effort = (effort or provider.reasoning_effort).strip().lower()
    is_xhigh = effective_effort == "xhigh"
    resolved_retries = max_retries
    if resolved_retries is None:
        resolved_retries = (
            XHIGH_RESPONSES_MAX_RETRIES
            if is_xhigh
            else DEFAULT_RESPONSES_MAX_RETRIES
        )
    resolved_timeout = timeout
    if resolved_timeout is None:
        resolved_timeout = (
            XHIGH_RESPONSES_TIMEOUT_SECONDS
            if is_xhigh
            else DEFAULT_RESPONSES_TIMEOUT_SECONDS
        )
    if resolved_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if resolved_timeout <= 0:
        raise ValueError("timeout must be positive")
    return int(resolved_retries), float(resolved_timeout)


def call_responses(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig | None = None,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
) -> LLMReply:
    """One ``/responses`` round-trip. ``instructions`` is the system prompt."""
    p = provider or get_provider()
    max_retries, timeout = _resolve_responses_request_policy(
        p, effort, max_retries, timeout
    )
    url = p.base_url.rstrip("/") + "/responses"
    payload = _responses_payload(
        instructions, user_input, p, effort, text_format
    )
    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {p.api_key}",
        "Content-Type": "application/json",
    }
    last = ""
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            text = _extract_output_text(data)
            if not text.strip():
                last = "empty completion"
            else:
                return LLMReply(text=text, model=p.model, raw=data)
        except urllib.error.HTTPError as e:
            # Read the body for the message but scrub nothing of ours leaks here.
            try:
                msg = e.read()[:300].decode("utf-8", "replace")
            except Exception:
                msg = ""
            last = f"HTTP {e.code}: {msg}"
            if e.code in (400, 401, 403, 404):
                break  # not retryable
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        if attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise ProviderError(f"provider call failed after {max_retries} tries: {last}")
