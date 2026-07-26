import hashlib
import json
import logging
import time
from collections.abc import Callable

import litellm

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    APPLY_PATCH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    max_tool_calls_per_response: int = 1
    enable_apply_patch_tool: bool = False
    compact_tool_history: bool = False
    compact_tool_arguments_over_chars: int = 4096
    compact_tool_outputs_keep_recent: int = 4


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        if not 1 <= self.config.max_tool_calls_per_response <= 4:
            raise ValueError("max_tool_calls_per_response must be in [1, 4]")
        if self.config.compact_tool_arguments_over_chars < 256:
            raise ValueError("compact_tool_arguments_over_chars must be at least 256")
        if self.config.compact_tool_outputs_keep_recent < 0:
            raise ValueError("compact_tool_outputs_keep_recent must be non-negative")
        self._physical_provider_calls = 0
        self._transport_retries = 0
        self._history_compaction_packets = 0
        self._history_compacted_items = 0
        self._history_compacted_chars = 0
        self._provider_attempts: list[dict] = []

    @property
    def physical_provider_calls(self) -> int:
        return self._physical_provider_calls

    @property
    def transport_retries(self) -> int:
        return self._transport_retries

    @property
    def history_compaction(self) -> dict[str, int | bool]:
        return {
            "enabled": self.config.compact_tool_history,
            "packets": self._history_compaction_packets,
            "items": self._history_compacted_items,
            "chars": self._history_compacted_chars,
        }

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
                    if message.get("name") == "apply_patch":
                        replacement = {
                            "path": (
                                parsed.get("path", "/output/candidate.json")
                                if isinstance(parsed, dict)
                                else "/output/candidate.json"
                            ),
                            "old_text": receipt,
                            "new_text": "",
                        }
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

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return self._compact_prepared_history(result)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                tools=[
                    BASH_TOOL_RESPONSE_API,
                    *(
                        [APPLY_PATCH_TOOL_RESPONSE_API]
                        if self.config.enable_apply_patch_tool
                        else []
                    ),
                ],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        prepared_messages = self._prepare_messages_for_api(messages)
        abort_exceptions = list(self.abort_exceptions)
        if not self.config.retry_timeouts:
            abort_exceptions.append(litellm.exceptions.Timeout)
        for attempt in retry(
            logger=logger,
            abort_exceptions=abort_exceptions,
            max_attempts=self.config.retry_attempts,
        ):
            with attempt:
                self._physical_provider_calls += 1
                if attempt.retry_state.attempt_number > 1:
                    self._transport_retries += 1
                started = time.monotonic()
                try:
                    response = self._query(prepared_messages, **kwargs)
                except Exception as exc:
                    self._provider_attempts.append(
                        {
                            "physical_call": self._physical_provider_calls,
                            "attempt": attempt.retry_state.attempt_number,
                            "elapsed_ms": (time.monotonic() - started) * 1000.0,
                            "status": "error",
                            "error_type": type(exc).__name__,
                        }
                    )
                    raise
                else:
                    self._provider_attempts.append(
                        {
                            "physical_call": self._physical_provider_calls,
                            "attempt": attempt.retry_state.attempt_number,
                            "elapsed_ms": (time.monotonic() - started) * 1000.0,
                            "status": "success",
                            "error_type": None,
                        }
                    )
        cost_output = self._calculate_cost(response)
        if self.config.use_global_model_stats:
            GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = self._parse_actions(response)
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

    def _parse_actions(self, response) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
            max_tool_calls=self.config.max_tool_calls_per_response,
            enable_apply_patch_tool=self.config.enable_apply_patch_tool,
        )

    def serialize(self) -> dict:
        value = super().serialize()
        info = value.setdefault("info", {})
        info["agent_runtime"] = {
            "protocol": "intentir.agent-runtime.v1",
            "history_compaction": self.history_compaction,
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
