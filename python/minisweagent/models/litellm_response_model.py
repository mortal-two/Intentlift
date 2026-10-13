import copy
import hashlib
import json
import logging
import signal
import time
import threading
from contextlib import contextmanager
from collections.abc import Callable

import litellm
import httpx

from minisweagent.exceptions import (
    FormatError,
    LimitsExceeded,
    MiniSWERequestCancelled,
)
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    APPLY_REPLAY_PATCH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API,
    MCP_CALL_TOOL_RESPONSE_API,
    MCP_FIRST_READ_TOOL_RESPONSE_API,
    SUBMIT_REPLAY_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    max_tool_calls_per_response: int = 1
    enable_bash_tool: bool = True
    enable_lift_mcp_tool: bool = False
    require_first_lift_mcp: bool = False
    routine_reasoning_effort: str | None = None
    routine_timeout_seconds: float | None = None
    semantic_initial_turns: int = 1
    enable_replay_patch_tool: bool = False
    enable_submit_replay_tool: bool = False
    compact_tool_history: bool = False
    compact_tool_arguments_over_chars: int = 4096
    compact_tool_outputs_keep_recent: int = 4
    compact_workspace_mirrors: bool = False
    compact_response_history_keep_recent: int = 0
    use_previous_response_id: bool = False
    # Transport recovery is deliberately separate from logical Lift calls.
    # The active workflow may disable retries for expensive Lift turns while
    # other callers retain a bounded transport-only retry.
    transient_retry_attempts: int = 0
    # Zero retains the generic model default. Replay sets both explicitly so a
    # changing source cannot turn into an unbounded provider-call episode.
    physical_provider_call_limit: int = 0
    transport_retry_limit: int = 0


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        if not 1 <= self.config.max_tool_calls_per_response <= 4:
            raise ValueError("max_tool_calls_per_response must be in [1, 4]")
        if self.config.compact_tool_arguments_over_chars < 256:
            raise ValueError("compact_tool_arguments_over_chars must be at least 256")
        if self.config.compact_tool_outputs_keep_recent < 0:
            raise ValueError("compact_tool_outputs_keep_recent must be non-negative")
        if self.config.compact_response_history_keep_recent < 0:
            raise ValueError(
                "compact_response_history_keep_recent must be non-negative"
            )
        if self.config.require_first_lift_mcp and not self.config.enable_lift_mcp_tool:
            raise ValueError(
                "require_first_lift_mcp needs enable_lift_mcp_tool"
            )
        if self.config.routine_reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise ValueError("unsupported routine_reasoning_effort")
        if (
            self.config.routine_timeout_seconds is not None
            and self.config.routine_timeout_seconds <= 0
        ):
            raise ValueError("routine_timeout_seconds must be positive when set")
        if not 0 <= self.config.semantic_initial_turns <= 4:
            raise ValueError("semantic_initial_turns must be in [0, 4]")
        if not 0 <= self.config.transient_retry_attempts <= 1:
            raise ValueError("transient_retry_attempts must be in [0, 1]")
        if self.config.physical_provider_call_limit < 0:
            raise ValueError("physical_provider_call_limit must be non-negative")
        if self.config.transport_retry_limit < 0:
            raise ValueError("transport_retry_limit must be non-negative")
        self._physical_provider_calls = 0
        self._transport_retries = 0
        self._history_compaction_packets = 0
        self._history_compacted_items = 0
        self._history_compacted_chars = 0
        self._workspace_mirror_compacted_blocks = 0
        self._workspace_mirror_compacted_chars = 0
        self._response_history_compacted_packets = 0
        self._response_history_compacted_items = 0
        self._response_history_compacted_chars = 0
        self._response_history_chain_identities: list[tuple[str, str]] = []
        self._response_history_chain_sha256 = ""
        self._response_history_chain_chars = 0
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._cache_write_input_tokens = 0
        self._provider_attempts: list[dict] = []
        self._continuation_requests = 0
        self._continuation_fallbacks = 0
        self._continuation_input_items = 0
        self._continuation_input_chars = 0
        self._stateless_input_items = 0
        self._stateless_input_chars = 0
        self._continuation_disabled = False
        self._continuation_disable_reason: str | None = None
        self._cancel_requested: Callable[[], int | None] | None = None

    def set_cancel_requested(
        self, callback: Callable[[], int | None] | None
    ) -> None:
        """Install a proof-irrelevant candidate-wave cancellation hook."""

        if callback is not None and not callable(callback):
            raise TypeError("cancel callback must be callable")
        self._cancel_requested = callback

    def _raise_if_cancelled(self) -> None:
        callback = self._cancel_requested
        winner = callback() if callback is not None else None
        if winner is not None:
            raise MiniSWERequestCancelled(winner)

    @property
    def physical_provider_calls(self) -> int:
        return self._physical_provider_calls

    @property
    def transport_retries(self) -> int:
        return self._transport_retries

    def _reserve_physical_provider_call(self, *, transport_retry: bool) -> None:
        """Reserve one actual request before it reaches the provider.

        The limit is intentionally checked at the physical attempt boundary,
        not merely before a logical model query.  This also caps a retry or a
        continuation fallback that happens inside a single query.
        """

        if (
            self.config.physical_provider_call_limit
            and self._physical_provider_calls
            >= self.config.physical_provider_call_limit
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "PhysicalProviderCallLimitExceeded",
                    "extra": {
                        "exit_status": "PhysicalProviderCallLimitExceeded",
                        "submission": "",
                        "physical_provider_calls": self._physical_provider_calls,
                        "physical_provider_call_limit": (
                            self.config.physical_provider_call_limit
                        ),
                    },
                }
            )
        if (
            transport_retry
            and self.config.transport_retry_limit
            and self._transport_retries >= self.config.transport_retry_limit
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "TransportRetryLimitExceeded",
                    "extra": {
                        "exit_status": "TransportRetryLimitExceeded",
                        "submission": "",
                        "transport_retries": self._transport_retries,
                        "transport_retry_limit": self.config.transport_retry_limit,
                    },
                }
            )
        self._physical_provider_calls += 1
        if transport_retry:
            self._transport_retries += 1

    @property
    def history_compaction(self) -> dict[str, int | bool]:
        return {
            "enabled": self.config.compact_tool_history,
            "packets": self._history_compaction_packets,
            "items": self._history_compacted_items,
            "chars": self._history_compacted_chars,
            "workspace_mirror_blocks": self._workspace_mirror_compacted_blocks,
            "workspace_mirror_chars": self._workspace_mirror_compacted_chars,
            "response_history_packets": self._response_history_compacted_packets,
            "response_history_items": self._response_history_compacted_items,
            "response_history_chars": self._response_history_compacted_chars,
        }

    @property
    def prompt_cache_telemetry(self) -> dict[str, int | str | bool]:
        key = self.config.model_kwargs.get("prompt_cache_key")
        return {
            "policy": (
                "stable-append-only-prefix"
                if not self.config.compact_tool_history
                and not self.config.compact_workspace_mirrors
                and self.config.compact_response_history_keep_recent == 0
                else "history-transforming"
            ),
            "key_present": isinstance(key, str) and bool(key),
            "input_tokens": self._input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "cache_write_input_tokens": self._cache_write_input_tokens,
        }

    @property
    def response_continuation_telemetry(self) -> dict[str, int | str | bool | None]:
        return {
            "enabled": self.config.use_previous_response_id,
            "disabled": self._continuation_disabled,
            "disable_reason": self._continuation_disable_reason,
            "requests": self._continuation_requests,
            "fallbacks": self._continuation_fallbacks,
            "continuation_input_items": self._continuation_input_items,
            "continuation_input_chars": self._continuation_input_chars,
            "stateless_input_items": self._stateless_input_items,
            "stateless_input_chars": self._stateless_input_chars,
        }

    @staticmethod
    def _usage_counter(value, key: str) -> int:
        observed = value.get(key) if isinstance(value, dict) else getattr(value, key, 0)
        return (
            observed
            if isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= 0
            else 0
        )

    @staticmethod
    def _compaction_receipt(*, kind: str, text: str) -> str:
        return json.dumps(
            {
                "protocol": "intentir.agent-history-compaction.v1",
                "kind": kind,
                "status": "compacted",
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
                "authority": (
                    "transport-only; rerun the tool or inspect the persistent "
                    "workspace for exact content"
                ),
            },
            sort_keys=True,
        )

    def _compact_prepared_history(self, messages: list[dict]) -> list[dict]:
        if not self.config.compact_tool_history:
            return messages
        compacted = [dict(message) for message in messages]
        output_indices = [
            index
            for index, message in enumerate(compacted)
            if message.get("type") == "function_call_output"
        ]
        keep = self.config.compact_tool_outputs_keep_recent
        stale_outputs = set(output_indices[:-keep] if keep else output_indices)
        packet_items = 0
        packet_chars = 0
        for index, message in enumerate(compacted):
            item_type = message.get("type")
            if item_type == "function_call":
                arguments = message.get("arguments")
                if (
                    isinstance(arguments, str)
                    and len(arguments)
                    > self.config.compact_tool_arguments_over_chars
                ):
                    receipt = self._compaction_receipt(
                        kind="tool_arguments", text=arguments
                    )
                    try:
                        parsed = json.loads(arguments)
                    except (TypeError, ValueError):
                        parsed = {}
                    if message.get("name") == "apply_replay_patch":
                        replacement = {
                            "path": (
                                parsed.get("path", "/output/candidate.json")
                                if isinstance(parsed, dict)
                                else "/output/candidate.json"
                            ),
                            "old_text": receipt,
                            "new_text": "",
                            "submit": False,
                        }
                    elif message.get("name") == "submit_replay":
                        replacement = {"source_text": receipt}
                    elif message.get("name") == "mcp_call_tool":
                        method = (
                            parsed.get("method", "resources/read")
                            if isinstance(parsed, dict)
                            else "resources/read"
                        )
                        params = (
                            parsed.get("params", {})
                            if isinstance(parsed, dict)
                            else {}
                        )
                        if not isinstance(params, dict):
                            params = {}
                        replacement = {"method": method, "params": params}
                    else:
                        replacement = {"command": f"# {receipt}"}
                    message["arguments"] = json.dumps(replacement, sort_keys=True)
                    packet_items += 1
                    packet_chars += max(
                        0, len(arguments) - len(message["arguments"])
                    )
            elif item_type == "function_call_output" and index in stale_outputs:
                output = message.get("output")
                if isinstance(output, str):
                    receipt = self._compaction_receipt(
                        kind="tool_output", text=output
                    )
                    message["output"] = receipt
                    packet_items += 1
                    packet_chars += max(0, len(output) - len(receipt))
        if packet_items:
            self._history_compaction_packets += 1
            self._history_compacted_items += packet_items
            self._history_compacted_chars += packet_chars
        return compacted

    @staticmethod
    def _replace_workspace_mirror_blocks(text: str) -> tuple[str, int, int]:
        """Replace re-readable exact workspace mirrors with deterministic receipts.

        The local trajectory is never changed.  Marked bytes already have an exact
        filesystem authority, so later stateless provider packets do not need to
        resend them after the model has received them once.
        """

        opening = "<persistent-workspace-mirror>"
        closing = "</persistent-workspace-mirror>"
        cursor = 0
        parts: list[str] = []
        blocks = 0
        removed_chars = 0
        while True:
            start = text.find(opening, cursor)
            if start < 0:
                parts.append(text[cursor:])
                break
            end = text.find(closing, start + len(opening))
            if end < 0:
                parts.append(text[cursor:])
                break
            exact_start = start + len(opening)
            exact_text = text[exact_start:end]
            receipt = json.dumps(
                {
                    "protocol": "intentir.agent-workspace-mirror.v1",
                    "status": "compacted-after-first-use",
                    "sha256": hashlib.sha256(exact_text.encode("utf-8")).hexdigest(),
                    "chars": len(exact_text),
                    "authority": (
                        "transport-only; exact bytes remain in the persistent "
                        "allowlisted workspace and may be reread"
                    ),
                },
                sort_keys=True,
            )
            parts.extend((text[cursor:exact_start], "\n", receipt, "\n"))
            cursor = end
            blocks += 1
            removed_chars += max(0, len(exact_text) - len(receipt) - 2)
        return "".join(parts), blocks, removed_chars

    def _compact_workspace_mirror_messages(
        self, messages: list[dict]
    ) -> list[dict]:
        if not self.config.compact_workspace_mirrors:
            return messages
        compacted = copy.deepcopy(messages)
        later_response = False
        response_after: list[bool] = [False] * len(compacted)
        for index in range(len(compacted) - 1, -1, -1):
            response_after[index] = later_response
            if compacted[index].get("object") == "response":
                later_response = True
        packet_blocks = 0
        packet_chars = 0
        for index, message in enumerate(compacted):
            if not response_after[index] or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                replaced, blocks, chars = self._replace_workspace_mirror_blocks(
                    content
                )
                message["content"] = replaced
                packet_blocks += blocks
                packet_chars += chars
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "input_text":
                        continue
                    text = item.get("text")
                    if not isinstance(text, str):
                        continue
                    replaced, blocks, chars = self._replace_workspace_mirror_blocks(
                        text
                    )
                    item["text"] = replaced
                    packet_blocks += blocks
                    packet_chars += chars
        self._workspace_mirror_compacted_blocks += packet_blocks
        self._workspace_mirror_compacted_chars += packet_chars
        return compacted

    def _compact_old_response_groups(self, messages: list[dict]) -> list[dict]:
        keep = self.config.compact_response_history_keep_recent
        if keep == 0:
            return messages
        response_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("object") == "response"
        ]
        if len(response_indices) <= keep:
            return messages
        cutoff = response_indices[-keep]
        dropped: list[dict] = []
        retained: list[dict] = []
        receipt_insert_at = 0
        for index, message in enumerate(messages):
            is_old_runtime_item = index < cutoff and (
                message.get("object") == "response"
                or message.get("type") == "function_call_output"
            )
            if is_old_runtime_item:
                dropped.append(message)
                receipt_insert_at = len(retained)
            else:
                retained.append(message)
        if not dropped:
            return messages
        identities = [self._response_history_item_identity(item) for item in dropped]
        cached = self._response_history_chain_identities
        if identities[: len(cached)] != cached:
            # Conversation replacement is legal only at explicit continuation
            # boundaries. Rebuild the transport-only chain once in that rare case;
            # the ordinary append-only path serializes each old item exactly once.
            cached = []
            self._response_history_chain_identities = []
            self._response_history_chain_sha256 = ""
            self._response_history_chain_chars = 0
        for identity, item in zip(
            identities[len(cached) :],
            dropped[len(cached) :],
            strict=True,
        ):
            encoded = json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
            previous = (
                bytes.fromhex(self._response_history_chain_sha256)
                if self._response_history_chain_sha256
                else bytes(32)
            )
            self._response_history_chain_sha256 = hashlib.sha256(
                previous + len(encoded).to_bytes(8, "big") + encoded
            ).hexdigest()
            self._response_history_chain_chars += len(encoded)
            self._response_history_chain_identities.append(identity)
        removed_chars = (
            self._response_history_chain_chars + max(0, len(dropped) - 1) + 2
        )
        receipt_text = json.dumps(
            {
                "protocol": "intentir.agent-response-history.v2",
                "status": "compacted",
                "dropped_items": len(dropped),
                "chain_sha256": self._response_history_chain_sha256,
                "canonical_item_chars": self._response_history_chain_chars,
                "authority": (
                    "transport-only; full trajectory is retained locally and exact "
                    "working files remain in the persistent workspace; chain is "
                    "sha256(previous_chain || item_length || canonical_item)"
                ),
            },
            sort_keys=True,
        )
        retained.insert(
            receipt_insert_at,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": receipt_text}],
            },
        )
        self._response_history_compacted_packets += 1
        self._response_history_compacted_items += len(dropped)
        self._response_history_compacted_chars += max(
            0, removed_chars - len(receipt_text)
        )
        return retained

    @staticmethod
    def _response_history_item_identity(message: dict) -> tuple[str, str]:
        """Return the append-only local identity of one runtime history item."""

        if message.get("object") == "response":
            value = message.get("id")
            if isinstance(value, str) and value:
                return ("response", value)
        if message.get("type") == "function_call_output":
            value = message.get("call_id")
            if isinstance(value, str) and value:
                return ("function_call_output", value)
        encoded = json.dumps(
            message,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return ("canonical", hashlib.sha256(encoded).hexdigest())

    def _compact_stale_lift_continuations(
        self, messages: list[dict]
    ) -> list[dict]:
        """Keep only the latest exact Gate1 continuation in the API packet.

        The full trajectory remains local.  The newest continuation contains the
        current exact feedback and candidate (or their persistent paths), so older
        continuation envelopes are stale transport state rather than authority.
        """

        marker = "intentir.lift-owner-continuation.v1"
        continuation_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
            and marker
            in json.dumps(message.get("content", ""), ensure_ascii=False, default=str)
        ]
        if len(continuation_indices) <= 1:
            return messages
        stale = set(continuation_indices[:-1])
        latest = continuation_indices[-1]
        dropped = [messages[index] for index in continuation_indices[:-1]]
        serialized = json.dumps(
            dropped,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        receipt_text = json.dumps(
            {
                "protocol": "intentir.lift-continuation-history.v1",
                "status": "stale-continuations-compacted",
                "dropped_items": len(dropped),
                "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                "chars": len(serialized),
                "authority": (
                    "transport-only; the full trajectory remains local and the "
                    "latest continuation is exact"
                ),
            },
            sort_keys=True,
        )
        retained: list[dict] = []
        for index, message in enumerate(messages):
            if index in stale:
                continue
            if index == latest:
                retained.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": receipt_text}],
                    }
                )
            retained.append(message)
        self._response_history_compacted_packets += 1
        self._response_history_compacted_items += len(dropped)
        self._response_history_compacted_chars += max(
            0, len(serialized) - len(receipt_text)
        )
        return retained

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        messages = self._compact_workspace_mirror_messages(messages)
        messages = self._compact_old_response_groups(messages)
        messages = self._compact_stale_lift_continuations(messages)
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return self._compact_prepared_history(result)

    @staticmethod
    def _latest_response_id(messages: list[dict]) -> tuple[int, str | None]:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("object") != "response":
                continue
            response_id = message.get("id")
            return index, response_id if isinstance(response_id, str) and response_id else None
        return -1, None

    @staticmethod
    def _packet_shape(messages: list[dict]) -> tuple[int, int]:
        return (
            len(messages),
            len(
                json.dumps(
                    messages,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            ),
        )

    @staticmethod
    def _is_previous_response_error(exc: Exception) -> bool:
        # A stateless replay of the exact local history is the authoritative
        # fallback. Once a continuation request itself receives a provider
        # BadRequest, retrying the same previous_response_id cannot add value
        # and only multiplies relay latency.
        if isinstance(exc, litellm.exceptions.BadRequestError):
            return True
        text = f"{type(exc).__name__}: {exc}".casefold()
        return (
            "previous_response_id" in text
            or "previous response id" in text
            or "no tool call found for function call output" in text
            or (
                "previous response" in text
                and any(token in text for token in ("not found", "unknown", "unsupported"))
            )
        )

    @staticmethod
    def _is_transient_provider_error(exc: BaseException) -> bool:
        """Identify provider failures safe for one transport-only retry.

        This deliberately excludes generic API errors and parsing/contract
        failures.  Those must remain fail-closed instead of consuming another
        physical provider call.  The explicit message check covers relays that
        wrap an EOF in a generic InternalServerError while retaining the
        transient transport signal.
        """

        transient_types = (
            litellm.exceptions.InternalServerError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.RateLimitError,
            httpx.TransportError,
        )
        if isinstance(exc, transient_types):
            return True
        text = f"{type(exc).__name__}: {exc}".casefold()
        return (
            "unexpected eof" in text
            or "stream ended without a completed response" in text
            or "responses stream emitted a failure event" in text
        )

    def _record_packet(self, messages: list[dict], *, continuation: bool) -> tuple[int, int]:
        items, chars = self._packet_shape(messages)
        if continuation:
            self._continuation_requests += 1
            self._continuation_input_items += items
            self._continuation_input_chars += chars
        else:
            self._stateless_input_items += items
            self._stateless_input_chars += chars
        return items, chars

    def _response_tools(
        self,
        *,
        response_count: int,
        response_tools: list[dict] | None = None,
    ) -> list[dict]:
        if response_tools is not None:
            if not response_tools or any(
                not isinstance(tool, dict) for tool in response_tools
            ):
                raise ValueError("dynamic Responses tool surface is malformed")
            return copy.deepcopy(response_tools)
        if self.config.require_first_lift_mcp and response_count == 0:
            # The first owner turn is deliberately read-only.  This makes the
            # MCP bridge an actual context boundary instead of an optional set
            # of tools that a provider can bypass by emitting a candidate from
            # the startup receipt alone.  Later turns receive the ordinary
            # MCP read/write surface and CandidateCheck remains authoritative.
            return [MCP_FIRST_READ_TOOL_RESPONSE_API]
        return [
            *(
                [BASH_TOOL_RESPONSE_API]
                if self.config.enable_bash_tool
                else []
            ),
            *([MCP_CALL_TOOL_RESPONSE_API] if self.config.enable_lift_mcp_tool else []),
            *(
                [APPLY_REPLAY_PATCH_TOOL_RESPONSE_API]
                if self.config.enable_replay_patch_tool
                else []
            ),
            *(
                [SUBMIT_REPLAY_TOOL_RESPONSE_API]
                if self.config.enable_submit_replay_tool
                else []
            ),
        ]

    @contextmanager
    def _hard_provider_deadline(self, timeout):
        """Enforce the configured turn deadline around streaming LiteLLM calls.

        LiteLLM's HTTP timeout does not reliably bound iteration over a relay's
        SSE stream.  Lift/Replay must still fail closed at the model-turn
        boundary, so the main-thread caller also owns a real wall-clock alarm.
        Worker-thread callers retain LiteLLM's native timeout because Python
        signals cannot safely interrupt a non-main thread.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or threading.current_thread() is not threading.main_thread()
        ):
            yield
            return
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()

        def on_alarm(_signum, _frame):
            raise litellm.exceptions.Timeout(
                "provider response exceeded the configured turn deadline",
                model=self.config.model_name,
                llm_provider="responses",
            )

        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            if previous_timer[0] > 0:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, previous_timer[0] - elapsed),
                    previous_timer[1],
                )
            else:
                signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    @staticmethod
    def _materialize_streaming_response(response):
        """Return the completed Responses object for a streaming LiteLLM call.

        A streaming Responses iterator is usable only after the provider emits
        its completed response envelope.  A closed function-call item is not a
        substitute for that envelope: accepting it would turn a truncated SSE
        transport into a valid Lift/Replay action.
        Duck typing is intentional: LiteLLM has changed the concrete iterator
        class between releases, while the completed-response contract is
        stable.
        """

        if not hasattr(response, "completed_response"):
            # ``_query`` always requests the Responses SSE path.  A relay may
            # nevertheless return a plain completed envelope, which is safe
            # only when it explicitly says ``status=completed`` and carries an
            # output list.  An arbitrary iterator or a function-call-looking
            # object is not a completion witness and must not reach the action
            # parser as if it were one.
            status = (
                response.get("status")
                if isinstance(response, dict)
                else getattr(response, "status", None)
            )
            output = (
                response.get("output")
                if isinstance(response, dict)
                else getattr(response, "output", None)
            )
            if status == "completed" and isinstance(output, (list, tuple)):
                return response
            raise RuntimeError(
                "Responses stream ended without a completed response object"
            )
        function_items: dict[str, dict] = {}
        anonymous_function_items: list[dict] = []
        for _event in response:
            event_type = getattr(_event, "type", None)
            if event_type in {"error", "response.failed"}:
                raise RuntimeError("Responses stream emitted a failure event")
            item = getattr(_event, "item", None)
            item_type = getattr(item, "type", None)
            if item_type != "function_call":
                continue
            item_value = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else dict(item)
            )
            item_id = item_value.get("id")
            if isinstance(item_id, str) and item_id:
                function_items[item_id] = item_value
            else:
                anonymous_function_items.append(item_value)
        completed_event = getattr(response, "completed_response", None)
        completed_response = getattr(completed_event, "response", None)
        if completed_response is None:
            raise RuntimeError(
                "Responses stream ended without a completed response object"
            )
        output = getattr(completed_response, "output", None)
        output_items = list(output) if isinstance(output, (list, tuple)) else []
        output_ids = {
            item_id
            for item in output_items
            if isinstance(
                item_id := (
                    item.get("id")
                    if isinstance(item, dict)
                    else getattr(item, "id", None)
                ),
                str,
            )
            and item_id
        }
        recovered_items = [
            item
            for item_id, item in function_items.items()
            if item_id not in output_ids
        ]
        recovered_items.extend(anonymous_function_items)
        if recovered_items:
            completed_response.output = [*output_items, *recovered_items]
        return completed_response

    def _query(
        self,
        messages: list[dict[str, str]],
        *,
        response_count: int = 0,
        semantic_turn: bool = False,
        response_tools: list[dict] | None = None,
        **kwargs,
    ):
        if self.config.routine_reasoning_effort is not None and not semantic_turn:
            kwargs["reasoning_effort"] = self.config.routine_reasoning_effort
        if self.config.routine_timeout_seconds is not None and not semantic_turn:
            configured_timeout = kwargs.get(
                "timeout", self.config.model_kwargs.get("timeout")
            )
            kwargs["timeout"] = (
                min(float(configured_timeout), self.config.routine_timeout_seconds)
                if isinstance(configured_timeout, (int, float))
                and not isinstance(configured_timeout, bool)
                and configured_timeout > 0
                else self.config.routine_timeout_seconds
            )
        try:
            model_kwargs = self.config.model_kwargs | kwargs
            # Materialize the SSE Responses stream into the completed-response
            # shape expected by action parsing.
            model_kwargs.setdefault("stream", True)
            tools = self._response_tools(
                response_count=response_count,
                response_tools=response_tools,
            )
            if tools:
                # Lift and Replay are action protocols.  A completed response
                # without a tool call cannot advance either protocol and only
                # creates an avoidable format-error turn.
                model_kwargs.setdefault("tool_choice", "required")
            if model_kwargs.get("timeout") is None:
                model_kwargs["timeout"] = httpx.Timeout(None)
            def issue_response(request_kwargs):
                response = litellm.responses(
                    model=self.config.model_name,
                    input=messages,
                    tools=tools,
                    **request_kwargs,
                )
                return self._materialize_streaming_response(response)

            with self._hard_provider_deadline(model_kwargs.get("timeout")):
                try:
                    return issue_response(model_kwargs)
                except AttributeError as exc:
                    # Some OpenAI-compatible Responses relays emit a valid
                    # streamed completion whose ``usage`` is a JSON mapping.
                    # Current LiteLLM then crashes in its *post-completion*
                    # logging callback while accessing ``response.usage`` as
                    # a pydantic field. The SSE request is already complete,
                    # so no partial output can be trusted. Reissue the same
                    # logical turn once through the completed Responses path,
                    # which preserves tools/strict JSON and still requires a
                    # full ``status=completed`` envelope below. Do not catch
                    # arbitrary AttributeError: those remain host failures.
                    if str(exc) != "'dict' object has no attribute 'usage'":
                        raise
                    completed_kwargs = dict(model_kwargs)
                    completed_kwargs["stream"] = False
                    return issue_response(completed_kwargs)
        except litellm.exceptions.AuthenticationError as e:
            e.message += " Configure credentials through the IntentIR provider settings."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response_tools = kwargs.pop("response_tools", None)
        if response_tools is not None and (
            not isinstance(response_tools, list)
            or any(not isinstance(tool, dict) for tool in response_tools)
        ):
            raise ValueError("dynamic Responses tool surface is malformed")
        last_response_index, previous_response_id = self._latest_response_id(messages)
        continuation_enabled = (
            self.config.use_previous_response_id
            and not self._continuation_disabled
            and previous_response_id is not None
            and last_response_index + 1 < len(messages)
        )
        if continuation_enabled:
            prepared_messages = self._prepare_messages_for_api(
                messages[last_response_index + 1 :]
            )
        else:
            prepared_messages = self._prepare_messages_for_api(messages)
            previous_response_id = None
        response_count = sum(
            message.get("object") == "response" for message in messages
        )
        last_response_index = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("object") == "response"
            ),
            default=-1,
        )
        continuation_turn = any(
            message.get("role") == "user"
            and "intentir.lift-owner-continuation.v1"
            in json.dumps(message.get("content", ""), ensure_ascii=False)
            for message in messages[last_response_index + 1 :]
        )
        semantic_turn = (
            response_count < self.config.semantic_initial_turns
            or continuation_turn
        )
        abort_exceptions = list(self.abort_exceptions)
        if not self.config.retry_timeouts:
            abort_exceptions.append(litellm.exceptions.Timeout)
        for attempt in retry(
            logger=logger,
            abort_exceptions=abort_exceptions,
            max_attempts=(
                self.config.retry_attempts
                + self.config.transient_retry_attempts
            ),
            retry_predicate=(
                self._is_transient_provider_error
                if self.config.transient_retry_attempts > 0
                else None
            ),
        ):
            # This check deliberately sits outside the retry attempt context:
            # cancellation is a candidate-wave decision, never a transport
            # failure eligible for another provider call.
            self._raise_if_cancelled()
            with attempt:
                call_was_fallback = False
                self._reserve_physical_provider_call(
                    transport_retry=attempt.retry_state.attempt_number > 1
                )
                started = time.monotonic()
                packet_items, packet_chars = self._record_packet(
                    prepared_messages,
                    continuation=previous_response_id is not None,
                )
                try:
                    response = self._query(
                        prepared_messages,
                        response_count=response_count,
                        semantic_turn=semantic_turn,
                        response_tools=response_tools,
                        **(
                            {"previous_response_id": previous_response_id}
                            if previous_response_id is not None
                            else {}
                        ),
                        **kwargs,
                    )
                except Exception as exc:
                    if (
                        previous_response_id is not None
                        and self._is_previous_response_error(exc)
                    ):
                        self._provider_attempts.append(
                            {
                                "physical_call": self._physical_provider_calls,
                                "attempt": attempt.retry_state.attempt_number,
                                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "reasoning_effort": (
                                    self.config.model_kwargs.get("reasoning_effort")
                                    if semantic_turn
                                    else self.config.routine_reasoning_effort
                                    or self.config.model_kwargs.get("reasoning_effort")
                                ),
                                "continuation": True,
                                "previous_response_id_present": True,
                                "fallback": False,
                                "input_items": packet_items,
                                "input_chars": packet_chars,
                            }
                        )
                        self._continuation_fallbacks += 1
                        self._continuation_disabled = True
                        self._continuation_disable_reason = type(exc).__name__
                        previous_response_id = None
                        prepared_messages = self._prepare_messages_for_api(messages)
                        call_was_fallback = True
                        self._reserve_physical_provider_call(transport_retry=False)
                        started = time.monotonic()
                        packet_items, packet_chars = self._record_packet(
                            prepared_messages, continuation=False
                        )
                        try:
                            response = self._query(
                                prepared_messages,
                                response_count=response_count,
                                semantic_turn=semantic_turn,
                                response_tools=response_tools,
                                **kwargs,
                            )
                        except Exception as fallback_exc:
                            exc = fallback_exc
                        else:
                            exc = None
                    if exc is None:
                        pass
                    else:
                        self._provider_attempts.append(
                            {
                                "physical_call": self._physical_provider_calls,
                                "attempt": attempt.retry_state.attempt_number,
                                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "reasoning_effort": (
                                    self.config.model_kwargs.get("reasoning_effort")
                                    if semantic_turn
                                    else self.config.routine_reasoning_effort
                                    or self.config.model_kwargs.get("reasoning_effort")
                                ),
                                "continuation": previous_response_id is not None,
                                "previous_response_id_present": (
                                    previous_response_id is not None
                                ),
                                "fallback": call_was_fallback,
                                "input_items": packet_items,
                                "input_chars": packet_chars,
                            }
                        )
                        raise exc
                if response is not None:
                    usage = (
                        response.get("usage")
                        if isinstance(response, dict)
                        else getattr(response, "usage", None)
                    )
                    input_details = (
                        usage.get("input_tokens_details")
                        if isinstance(usage, dict)
                        else getattr(usage, "input_tokens_details", None)
                    )
                    input_tokens = self._usage_counter(usage, "input_tokens")
                    cached_input_tokens = self._usage_counter(
                        input_details, "cached_tokens"
                    ) or self._usage_counter(usage, "cached_input_tokens")
                    cache_write_input_tokens = self._usage_counter(
                        input_details, "cache_write_tokens"
                    ) or self._usage_counter(usage, "cache_write_input_tokens")
                    self._input_tokens += input_tokens
                    self._cached_input_tokens += cached_input_tokens
                    self._cache_write_input_tokens += cache_write_input_tokens
                    self._provider_attempts.append(
                        {
                            "physical_call": self._physical_provider_calls,
                            "attempt": attempt.retry_state.attempt_number,
                            "elapsed_ms": (time.monotonic() - started) * 1000.0,
                            "status": "success",
                            "error_type": None,
                            "input_tokens": input_tokens,
                            "cached_input_tokens": cached_input_tokens,
                            "cache_write_input_tokens": cache_write_input_tokens,
                            "reasoning_effort": (
                                self.config.model_kwargs.get("reasoning_effort")
                                if semantic_turn
                                else self.config.routine_reasoning_effort
                                or self.config.model_kwargs.get("reasoning_effort")
                            ),
                            "continuation": previous_response_id is not None,
                            "previous_response_id_present": (
                                previous_response_id is not None
                            ),
                            "fallback": call_was_fallback,
                            "input_items": packet_items,
                            "input_chars": packet_chars,
                        }
                    )
        # Preserve the completed provider-attempt receipt above, but do not let
        # a late response start another tool/model step after a sibling won.
        self._raise_if_cancelled()
        cost_output = self._calculate_cost(response)
        if self.config.use_global_model_stats:
            GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = self._parse_actions(
                response,
                response_count=response_count,
            )
        except FormatError as e:
            # hasattr guard: litellm.responses() returns a pydantic object, but tests
            # may inject a plain dict; dict(response) is the correct fallback in that case.
            # Inner try: if serialization itself fails, repr() guarantees the key is always set.
            try:
                e.messages[0]["extra"]["response"] = (
                    response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
                )
            except Exception:
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": actions,
            **cost_output,
            "timestamp": time.time(),
            "provider_elapsed_ms": self._provider_attempts[-1]["elapsed_ms"],
        }
        return message

    def _parse_actions(self, response, *, response_count: int = 0) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
            max_tool_calls=self.config.max_tool_calls_per_response,
            enable_bash_tool=self.config.enable_bash_tool,
            enable_lift_mcp_tool=self.config.enable_lift_mcp_tool,
            mcp_first_read_only=(
                self.config.require_first_lift_mcp
                and response_count == 0
            ),
            enable_replay_patch_tool=self.config.enable_replay_patch_tool,
            enable_submit_replay_tool=self.config.enable_submit_replay_tool,
        )

    def serialize(self) -> dict:
        value = super().serialize()
        info = value.setdefault("info", {})
        info["agent_runtime"] = {
            "protocol": "intentir.agent-runtime.v1",
            "history_compaction": self.history_compaction,
            "prompt_cache": self.prompt_cache_telemetry,
            "response_continuation": self.response_continuation_telemetry,
            "provider_attempts": [dict(item) for item in self._provider_attempts],
        }
        return value

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
