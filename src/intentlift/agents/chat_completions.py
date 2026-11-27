"""OpenAI-compatible Chat Completions with complete tool-call round trips."""
import json
import os
from .host import TOOLS
from .protocol import ToolCall, Turn, arguments, tool_loop
from ..core.errors import LiftError, require


class ChatCompletionsBackend:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def connect(self):
        from openai import OpenAI
        require(self.model, 'set --model or OPENAI_MODEL', 'configuration_failure')
        require(os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY required', 'configuration_failure')
        self.client = OpenAI(max_retries=0, timeout=float(os.environ.get('LIFT_API_TIMEOUT', '600')))

    def start(self, instructions, context):
        self.connect()
        self.history = [{'role': 'system', 'content': instructions},
                        {'role': 'user', 'content': json.dumps(context)}]

    def request(self, messages):
        from openai import OpenAIError
        self.calls += 1
        tools = [{'type': 'function', 'function': {k: v for k, v in tool.items() if k != 'type'}} for tool in TOOLS]
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           tools=tools, parallel_tool_calls=False)
        except OpenAIError as exc:
            raise LiftError('api_failure', 'Chat Completions request failed: ' + type(exc).__name__, api_calls=self.calls) from None
        require(response.choices and response.choices[0].finish_reason in {'stop', 'tool_calls'},
                'Chat completion is incomplete or rejected', 'api_failure')
        message = response.choices[0].message
        wire = {'role': 'assistant', 'content': message.content,
                **({'tool_calls': [c.model_dump(exclude_none=True) for c in message.tool_calls]} if message.tool_calls else {})}
        calls = tuple(ToolCall(c.id, c.function.name, arguments(c.function.arguments)) for c in message.tool_calls or [])
        return wire, Turn(message.content or '', calls)

    def next_turn(self):
        wire, turn = self.request(self.history)
        self.history.append(wire)
        return turn

    def observe(self, results):
        self.history.extend({'role': 'tool', 'tool_call_id': call.call_id,
                             'content': json.dumps(result, allow_nan=False)} for call, result in results)

    def feedback(self, result):
        self.history.append({'role': 'user', 'content': json.dumps(result)})

    def run(self, host, failure):
        try:
            return tool_loop(host, self)
        finally:
            if hasattr(self, 'client'):
                self.client.close()
