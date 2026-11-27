"""OpenAI-compatible Responses API; the default invocation backend."""
import json
import os
from .protocol import ToolCall, Turn, arguments, tool_loop
from .host import TOOLS
from ..core.errors import LiftError, require


class ResponsesBackend:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def start(self, instructions, context):
        from openai import OpenAI
        require(self.model, 'set --model or OPENAI_MODEL', 'configuration_failure')
        require(os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY required', 'configuration_failure')
        self.client = OpenAI(max_retries=0, timeout=float(os.environ.get('LIFT_API_TIMEOUT', '600')))
        self.instructions = instructions
        self.history = [{'role': 'user', 'content': json.dumps(context)}]

    def next_turn(self):
        from openai import OpenAIError
        self.calls += 1
        try:
            response = self.client.responses.create(model=self.model, instructions=self.instructions,
                    tools=TOOLS, parallel_tool_calls=False, input=self.history)
        except OpenAIError as exc:
            raise LiftError('api_failure', 'Responses request failed: ' + type(exc).__name__, api_calls=self.calls) from None
        require(response.status == 'completed', 'Responses completion is incomplete', 'api_failure')
        self.history.extend(item.model_dump(exclude_none=True) for item in response.output)
        calls = tuple(ToolCall(item.call_id, item.name, arguments(item.arguments))
                      for item in response.output if item.type == 'function_call')
        return Turn(response.output_text, calls)

    def observe(self, results):
        self.history.extend({'type': 'function_call_output', 'call_id': call.call_id,
                             'output': json.dumps(result, allow_nan=False)} for call, result in results)

    def feedback(self, result):
        self.history.append({'role': 'user', 'content': json.dumps(result)})

    def run(self, host, failure):
        try:
            return tool_loop(host, self)
        finally:
            if hasattr(self, 'client'):
                self.client.close()
