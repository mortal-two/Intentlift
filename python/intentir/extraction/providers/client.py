"""Minimal HTTP client for the LLM provider (OpenAI ``responses`` wire API).

Dependency-free (stdlib ``urllib``) so extraction runs without extra packages.
The API key is read from :mod:`provider` and is NEVER logged, printed, or put in
an exception message — only ``Authorization`` header bytes that urllib sends.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .provider import ProviderConfig, get_provider


# Source-aware Lift and source-blind Replay each own their logical retry
# budget. The default Responses transport allows one additional physical
# attempt only for a transient transport/5xx response; it never turns an empty
# or structurally invalid completion into an implicit second logical Lift.
# Explicit callers may still supply a bounded policy.
DEFAULT_RESPONSES_MAX_RETRIES = 2
DEFAULT_RESPONSES_TIMEOUT_SECONDS: float | None = None
XHIGH_RESPONSES_MAX_RETRIES = 2
XHIGH_RESPONSES_TIMEOUT_SECONDS: float | None = None
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_OVERLOADED_RESPONSE_TEXTS = frozenset(
    {
        "we're currently experiencing high demand, which may cause temporary errors.",
        "we’re currently experiencing high demand, which may cause temporary errors.",
    }
)
# A missing timeout is intentionally unbounded.  A positive explicit timeout
# remains available to callers that own a local experiment or compiler budget.
_DEFAULT_RESPONSES_TIMEOUT = object()


class ProviderError(RuntimeError):
    """Provider failure with secret-free request accounting evidence."""

    def __init__(
        self,
        message: str,
        *,
        request_sha256: str,
        attempts: int,
        elapsed_ms: float,
        failure_class: str = "provider_transport_error",
    ) -> None:
        super().__init__(message)
        self.request_sha256 = request_sha256
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        if not isinstance(failure_class, str) or not failure_class:
            raise ValueError("provider failure_class must be non-empty")
        self.failure_class = failure_class


@dataclass
class LLMReply:
    text: str
    model: str
    raw: dict
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = b""
    attempts: int = 1
    elapsed_ms: float = 0.0
    # A tool-only Responses completion is meaningful to the transport health
    # probe but not to the ordinary text-only Lift caller.  Keep this explicit
    # rather than treating a closed function-call item as a stream witness.
    stream_completed: bool = False


def _content_text_chunks(value: Any) -> list[str]:
    """Extract text items without assuming one provider's response model."""

    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    chunks: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") not in (None, "output_text", "text"):
            continue
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def _extract_output_text(data: dict) -> str:
    """Pull assistant text from Responses and compatible response envelopes."""

    # Preferred: the SDK convenience field.
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") not in (None, "message"):
                continue
            chunks.extend(_content_text_chunks(item.get("content")))
    if chunks:
        return "".join(chunks)

    # A few OpenAI-compatible relays expose a Chat Completions-shaped envelope
    # even when the endpoint is selected as Responses.  Supporting that shape
    # here prevents a valid model answer from becoming a false empty-completion
    # error; the caller still applies the exact SCRC JSON contract afterwards.
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping):
                chunks.extend(_content_text_chunks(message.get("content")))
            elif isinstance(choice.get("text"), str):
                chunks.append(choice["text"])
    return "".join(chunks)


def _empty_completion_diagnostic(data: Mapping[str, Any]) -> str:
    """Describe an empty model completion without retaining response content."""

    details: list[str] = ["empty completion"]
    status = data.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    output = data.get("output")
    if isinstance(output, list):
        item_types = [
            item.get("type")
            for item in output
            if isinstance(item, Mapping) and isinstance(item.get("type"), str)
        ]
        if item_types:
            details.append("output_types=" + ",".join(item_types[:8]))
    choices = data.get("choices")
    if isinstance(choices, list):
        details.append(f"choices={len(choices)}")
    return "; ".join(details)


def _provider_text_failure(text: str) -> str | None:
    """Classify an exact provider-owned plaintext failure completion.

    Responses-compatible relays sometimes return a successful HTTP/SSE
    envelope whose entire assistant output is a service-status message.  Such
    text is not a model-authored candidate.  Match only a closed whole-response
    set so ordinary candidate strings containing similar words remain under
    CandidateCheck authority.
    """

    normalized = " ".join(text.split()).casefold()
    if normalized in _OVERLOADED_RESPONSE_TEXTS:
        return "provider_overloaded"
    return None


def _responses_payload(
    instructions: str,
    user_input: str,
    provider: ProviderConfig,
    effort: str | None,
    text_format: Mapping[str, Any] | None,
    max_output_tokens: int | None,
    tools: list[Mapping[str, Any]] | None,
    tool_choice: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "instructions": instructions,
        # Keep one canonical typed message rather than relying on a
        # transport-specific string shortcut.
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_input},
                ],
            }
        ],
        "stream": True,
        "reasoning": {"effort": effort or provider.reasoning_effort},
    }
    if text_format is not None:
        payload["text"] = {"format": dict(text_format)}
    if max_output_tokens is not None:
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        payload["max_output_tokens"] = max_output_tokens
    if tools is not None:
        if not isinstance(tools, list) or not tools or any(
            not isinstance(tool, Mapping) for tool in tools
        ):
            raise ValueError("tools must be a non-empty list of mappings")
        payload["tools"] = [dict(tool) for tool in tools]
    if tool_choice is not None:
        if not isinstance(tool_choice, (str, Mapping)):
            raise TypeError("tool_choice must be a string, mapping, or None")
        payload["tool_choice"] = (
            dict(tool_choice) if isinstance(tool_choice, Mapping) else tool_choice
        )
    return payload


def responses_request_body(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
    tools: list[Mapping[str, Any]] | None = None,
    tool_choice: Mapping[str, Any] | str | None = None,
) -> bytes:
    """Return the exact secret-free JSON bytes sent to ``/responses``."""

    return json.dumps(
        _responses_payload(
            instructions,
            user_input,
            provider,
            effort,
            text_format,
            max_output_tokens,
            tools,
            tool_choice,
        ),
        separators=(",", ":"),
    ).encode()


def responses_request_sha256(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
    tools: list[Mapping[str, Any]] | None = None,
    tool_choice: Mapping[str, Any] | str | None = None,
) -> str:
    return hashlib.sha256(
        responses_request_body(
            instructions,
            user_input,
            provider=provider,
            effort=effort,
            text_format=text_format,
            max_output_tokens=max_output_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
    ).hexdigest()


def _normalize_stream_response(raw_response: bytes) -> tuple[dict[str, Any], str]:
    """Normalize a Responses SSE body into the existing reply envelope.

    Some OpenAI-compatible Responses relays expose the final response object
    with an empty ``output`` list and publish the assistant text only through
    ``response.output_text.delta`` events.  Preserve the final response
    metadata and synthesize the normal output text field from those deltas so
    Lift and Replay keep one response contract regardless of transport mode.
    """

    event_payloads: list[dict[str, Any]] = []
    pending_data: list[str] = []

    def consume_pending() -> None:
        if not pending_data:
            return
        raw_data = "\n".join(pending_data).strip()
        pending_data.clear()
        if not raw_data or raw_data == "[DONE]":
            return
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ValueError("Responses SSE contained invalid JSON") from exc
        if isinstance(payload, Mapping):
            event_payloads.append(dict(payload))

    for raw_line in raw_response.splitlines():
        line = raw_line.decode("utf-8", errors="replace")
        if line == "":
            consume_pending()
            continue
        if line.startswith("data:"):
            pending_data.append(line[5:].lstrip())
    consume_pending()

    if not event_payloads:
        raise ValueError("Responses SSE contained no data events")

    final_response: dict[str, Any] | None = None
    function_items: dict[str, dict[str, Any]] = {}
    anonymous_function_items: list[dict[str, Any]] = []
    delta_chunks: list[str] = []
    completed_text: str | None = None
    saw_completed = False
    for payload in event_payloads:
        event_type = payload.get("type")
        if event_type in {"error", "response.failed"}:
            raise ValueError("Responses SSE emitted a failure event")
        response = payload.get("response")
        if isinstance(response, Mapping):
            final_response = dict(response)
        item = payload.get("item")
        if isinstance(item, Mapping) and item.get("type") == "function_call":
            normalized_item = dict(item)
            item_id = normalized_item.get("id")
            if isinstance(item_id, str) and item_id:
                function_items[item_id] = normalized_item
            else:
                anonymous_function_items.append(normalized_item)
        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                delta_chunks.append(delta)
        elif event_type == "response.output_text.done":
            text = payload.get("text")
            if isinstance(text, str):
                completed_text = text
        elif event_type == "response.completed":
            saw_completed = True

    if not saw_completed:
        raise ValueError("Responses SSE ended before response.completed")

    if final_response is None:
        final_response = {}
    output = final_response.get("output")
    output_items = list(output) if isinstance(output, list) else []
    output_ids = {
        item.get("id")
        for item in output_items
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    recovered_function_items = [
        item for item_id, item in function_items.items() if item_id not in output_ids
    ]
    recovered_function_items.extend(anonymous_function_items)
    if recovered_function_items:
        final_response["output"] = [*output_items, *recovered_function_items]
    text = _extract_output_text(final_response)
    if not text:
        text = completed_text or "".join(delta_chunks)
    if text:
        # Keep both compatibility forms: output_text is preferred by the
        # extractor, while output/message/content remains a normal Responses
        # envelope for downstream provenance readers.
        final_response["output_text"] = text
        output = final_response.get("output")
        if not isinstance(output, list) or not output:
            final_response["output"] = [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ]
    return final_response, text


def _read_completed_sse_response(response: Any) -> bytes:
    """Read an SSE response through its protocol completion event.

    A Responses relay is permitted to keep an HTTP stream alive after it has
    emitted ``response.completed``.  ``HTTPResponse.read()`` waits for EOF in
    that case, which turns a completed provider turn into an unbounded local
    wait.  Consume the wire incrementally and close our side immediately once
    the protocol's own terminal event arrives.  This is deliberately not a
    wall-clock deadline and does not limit model effort, output, or retries.

    The returned bytes retain the original SSE framing up to and including the
    terminal event, so the normal source-blind parser remains the sole
    authority for validating the response envelope.
    """

    chunks: list[bytes] = []
    pending_data: list[bytes] = []
    while True:
        line = response.readline()
        if not line:
            return b"".join(chunks)
        chunks.append(line)
        normalized = line.rstrip(b"\r\n")
        if normalized.startswith(b"data:"):
            pending_data.append(normalized[5:].lstrip())
            continue
        if normalized:
            continue
        if not pending_data:
            continue
        event_data = b"\n".join(pending_data).strip()
        pending_data.clear()
        if event_data == b"[DONE]":
            return b"".join(chunks)
        try:
            event = json.loads(event_data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Preserve malformed data for the established parser to classify;
            # it must not become a transport-side partial candidate.
            continue
        if isinstance(event, Mapping) and event.get("type") == "response.completed":
            return b"".join(chunks)


def _resolve_responses_request_policy(
    provider: ProviderConfig,
    effort: str | None,
    max_retries: int | None,
    timeout: float | None | object,
) -> tuple[int, float | None]:
    """Resolve the physical transport policy for one Lift/Replay turn.

    Provider transport has no implicit wall-time deadline.  This keeps the
    provider layer from imposing a hidden effort-dependent ceiling; lifecycle
    owners may still pass an explicit positive timeout.
    Explicit caller values always win.
    """
    effective_effort = (effort or provider.reasoning_effort).strip().lower()
    is_extended_effort = effective_effort in {"xhigh", "max"}
    resolved_retries = max_retries
    if resolved_retries is None:
        resolved_retries = (
            XHIGH_RESPONSES_MAX_RETRIES
            if is_extended_effort
            else DEFAULT_RESPONSES_MAX_RETRIES
        )
    if timeout is _DEFAULT_RESPONSES_TIMEOUT:
        resolved_timeout = None
    elif timeout is None:
        resolved_timeout = None
    else:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError(
                "timeout must be None or a finite positive number"
            )
        resolved_timeout = float(timeout)
    if resolved_retries < 1:
        raise ValueError("max_retries must be at least 1")
    return int(resolved_retries), resolved_timeout


def call_responses(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig | None = None,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_retries: int | None = None,
    timeout: float | None | object = _DEFAULT_RESPONSES_TIMEOUT,
    max_output_tokens: int | None = None,
    tools: list[Mapping[str, Any]] | None = None,
    tool_choice: Mapping[str, Any] | str | None = None,
    allow_empty_output: bool = False,
) -> LLMReply:
    """Run one logical streaming ``/responses`` Lift/Replay turn.

    A transient transport/5xx failure may consume one additional physical
    request under the default policy.  The caller still counts this as one
    logical Lift or Replay turn and records the physical attempt count.
    """
    p = provider or get_provider()
    max_retries, timeout = _resolve_responses_request_policy(
        p, effort, max_retries, timeout
    )
    url = p.base_url.rstrip("/") + "/responses"
    body = responses_request_body(
        instructions,
        user_input,
        provider=p,
        effort=effort,
        text_format=text_format,
        max_output_tokens=max_output_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )
    request_sha256 = hashlib.sha256(body).hexdigest()
    headers = {
        "Authorization": f"Bearer {p.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    last = ""
    failure_class = "provider_transport_error"
    started = time.monotonic()
    attempts_made = 0
    for attempt in range(max_retries):
        attempts_made = attempt + 1
        retryable = False
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                content_type = r.headers.get("Content-Type", "").casefold()
                if "text/event-stream" in content_type:
                    raw_response = _read_completed_sse_response(r)
                else:
                    raw_response = r.read()
            stream_completed = False
            if "text/event-stream" in content_type or b"data:" in raw_response:
                data, text = _normalize_stream_response(raw_response)
                stream_completed = True
            else:
                data = json.loads(raw_response)
                if not isinstance(data, dict):
                    last = "Responses payload is not a JSON object"
                    failure_class = "provider_protocol_error"
                    break
                text = _extract_output_text(data)
            if not isinstance(data, dict):
                last = "Responses payload is not a JSON object"
                failure_class = "provider_protocol_error"
                break
            text_failure = _provider_text_failure(text)
            if text_failure is not None:
                last = "provider returned a service-overload completion"
                failure_class = text_failure
                retryable = True
            elif not text.strip() and not allow_empty_output:
                last = _empty_completion_diagnostic(data)
                failure_class = "provider_no_candidate"
            else:
                return LLMReply(
                    text=text,
                    model=p.model,
                    raw=data,
                    request_sha256=request_sha256,
                    response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    raw_response_sha256=hashlib.sha256(raw_response).hexdigest(),
                    raw_response=raw_response,
                    attempts=attempts_made,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                    stream_completed=stream_completed,
                )
        except json.JSONDecodeError:
            last = "Responses endpoint returned invalid JSON"
            failure_class = "provider_protocol_error"
            break
        except ValueError as e:
            # Stream-level failures are provider preparation failures.  Do not
            # let the parser exception escape as an unclassified local error,
            # and do not turn a partial SSE response into a candidate.
            message = str(e)
            transient_stream_failure = (
                "failure event" in message
                or "response.completed" in message
            )
            last = (
                "Responses SSE emitted a failure event"
                if "failure event" in message
                else "Responses SSE ended before response.completed"
                if "response.completed" in message
                else "Responses SSE protocol error"
            )
            failure_class = (
                "provider_transport_error"
                if transient_stream_failure
                else "provider_protocol_error"
            )
            retryable = transient_stream_failure
        except urllib.error.HTTPError as e:
            # Do not retain the provider body: relift diagnostics must remain
            # secret-free and must never echo a relay's prompt/error payload.
            last = f"HTTP {e.code}"
            failure_class = (
                "provider_authentication_error"
                if e.code in (401, 403)
                else "provider_rate_limit"
                if e.code == 429
                else "provider_timeout"
                if e.code == 408
                else "provider_http_error"
            )
            retryable = e.code in _RETRYABLE_HTTP_STATUS
            if not retryable:
                break  # not retryable
        except (urllib.error.URLError, TimeoutError) as e:
            # Keep endpoint details and relay-provided text out of the closed
            # diagnostic.  The typed class below is sufficient for lifecycle
            # handling and avoids leaking transport metadata into evidence.
            last = type(e).__name__
            reason = getattr(e, "reason", None)
            is_timeout = (
                isinstance(e, TimeoutError)
                or isinstance(reason, TimeoutError)
                or "timeout" in type(reason).__name__.casefold()
            )
            failure_class = (
                "provider_timeout"
                if is_timeout
                else "provider_transport_error"
            )
            retryable = True
        if retryable and attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
        elif not retryable:
            break
    raise ProviderError(
        f"provider call failed after {attempts_made} tries: {last}",
        request_sha256=request_sha256,
        attempts=attempts_made,
        elapsed_ms=(time.monotonic() - started) * 1000.0,
        failure_class=failure_class,
    )


def probe_responses_strict_json_schema(
    provider: ProviderConfig | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Probe strict JSON Schema support without exposing provider secrets.

    The probe deliberately uses a tiny closed schema rather than the full SCRC
    schema.  ``request_accepted`` answers whether the selected Responses
    endpoint accepted ``strict: true``; ``response_valid_json`` separately
    records whether the returned text satisfied the tiny object contract.  No
    response body, endpoint URL, API key, or provider error text is retained.
    """

    p = provider or get_provider()
    text_format = {
        "type": "json_schema",
        "name": "intentir_strict_schema_probe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    started = time.monotonic()
    try:
        reply = call_responses(
            "Return the JSON object required by the supplied schema.",
            "Set ok to true.",
            provider=p,
            effort=p.reasoning_effort,
            text_format=text_format,
            max_retries=1,
            timeout=timeout,
            max_output_tokens=32,
        )
    except ProviderError as exc:
        if exc.failure_class == "provider_no_candidate":
            return {
                "status": "accepted_no_response",
                "request_accepted": True,
                "response_valid_json": False,
                "failure_class": exc.failure_class,
                "attempts": exc.attempts,
                "elapsed_ms": exc.elapsed_ms,
            }
        status = (
            "rejected"
            if exc.failure_class in {
                "provider_http_error",
                "provider_rate_limit",
                "provider_protocol_error",
                "provider_authentication_error",
            }
            else "transport_error"
        )
        return {
            "status": status,
            "request_accepted": False,
            "response_valid_json": False,
            "failure_class": exc.failure_class,
            "attempts": exc.attempts,
            "elapsed_ms": exc.elapsed_ms,
        }

    parsed: Any = None
    response_valid_json = False
    try:
        parsed = json.loads(reply.text)
        response_valid_json = (
            isinstance(parsed, dict)
            and set(parsed) == {"ok"}
            and isinstance(parsed.get("ok"), bool)
        )
    except (TypeError, ValueError, RecursionError):
        response_valid_json = False
    return {
        "status": "accepted" if response_valid_json else "accepted_invalid_response",
        "request_accepted": True,
        "response_valid_json": response_valid_json,
        "failure_class": None,
        "attempts": reply.attempts,
        "elapsed_ms": reply.elapsed_ms,
        "probe_elapsed_ms": (time.monotonic() - started) * 1000.0,
    }


def probe_responses_tool_stream_completion(
    provider: ProviderConfig | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Verify one completed SSE Responses tool call without retaining content.

    This is a transport prerequisite only.  It deliberately verifies the
    completed stream envelope, the recovered final function-call item, and a
    tiny closed receipt; it has no CandidateCheck, TypeCheck, Gate, or closure
    authority.  The return value is safe to persist in evidence because it
    contains no endpoint, provider body, prompt, arguments, or credential.
    """

    p = provider or get_provider()
    function_name = "intentir_transport_probe_ack"
    tool = {
        "type": "function",
        "name": function_name,
        "description": "Acknowledge a completed Responses tool stream.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    started = time.monotonic()
    base = {
        "request_accepted": False,
        "stream_completed": False,
        "function_call_received": False,
        "function_receipt_valid": False,
    }
    try:
        reply = call_responses(
            "Use the required tool exactly once. Do not emit text.",
            "Call the tool with ok set to true.",
            provider=p,
            effort=p.reasoning_effort,
            tools=[tool],
            tool_choice={"type": "function", "name": function_name},
            max_retries=1,
            timeout=timeout,
            max_output_tokens=32,
            allow_empty_output=True,
        )
    except ProviderError as exc:
        status = (
            "rejected"
            if exc.failure_class
            in {
                "provider_http_error",
                "provider_rate_limit",
                "provider_protocol_error",
                "provider_authentication_error",
            }
            else "transport_error"
        )
        return {
            "status": status,
            **base,
            "failure_class": exc.failure_class,
            "attempts": exc.attempts,
            "elapsed_ms": exc.elapsed_ms,
        }

    output = reply.raw.get("output") if isinstance(reply.raw, Mapping) else None
    output_items = output if isinstance(output, list) else []
    function_items = [
        item
        for item in output_items if isinstance(item, Mapping)
        and item.get("type") == "function_call"
        and item.get("name") == function_name
    ]
    receipt_valid = False
    if len(function_items) == 1:
        arguments = function_items[0].get("arguments")
        try:
            receipt_valid = json.loads(arguments) == {"ok": True}
        except (TypeError, ValueError, RecursionError):
            receipt_valid = False
    result = {
        "status": (
            "accepted"
            if reply.stream_completed and receipt_valid
            else "accepted_invalid_completion"
        ),
        "request_accepted": True,
        "stream_completed": reply.stream_completed,
        "function_call_received": bool(function_items),
        "function_receipt_valid": receipt_valid,
        "failure_class": None,
        "attempts": reply.attempts,
        "elapsed_ms": reply.elapsed_ms,
        "probe_elapsed_ms": (time.monotonic() - started) * 1000.0,
    }
    return result
