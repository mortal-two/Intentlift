"""Actual MiniSWE controller using a restricted model and environment protocol."""
import json
import os
from types import SimpleNamespace
from .chat_completions import ChatCompletionsBackend
from ..core.errors import LiftError


class MiniSweBackend:
    def __init__(self, model):
        self.model = model

    def run(self, host, failure):
        settings = {'MSWEA_SILENT_STARTUP': '1',
                    'MSWEA_GLOBAL_CONFIG_DIR': str(host.work / 'mini-config')}
        previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)
        try:
            return self._run(host, failure)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _run(self, host, failure):
        try:
            from minisweagent.agents.default import DefaultAgent
        except ImportError as exc:
            raise LiftError('configuration_failure', 'install intentlift[miniswe] for MiniSWE') from exc
        transport = ChatCompletionsBackend(self.model)
        transport.connect()

        class Environment:
            config = SimpleNamespace()

            def execute(self, action, cwd=''):
                return host.invoke(action['name'], action['arguments'])

            def get_template_vars(self):
                return {}

            def serialize(self):
                return {}

        class Model:
            config = SimpleNamespace()

            def query(self, messages):
                wire = [{k: v for k, v in message.items() if k != 'extra'} for message in messages if message['role'] != 'exit']
                response, turn = transport.request(wire)
                response['extra'] = {'actions': [{'name': c.name, 'arguments': c.arguments, 'call_id': c.call_id} for c in turn.calls]}
                return response

            def format_message(self, **kwargs):
                return kwargs

            def format_observation_messages(self, message, outputs, template_vars=None):
                actions = message.get('extra', {}).get('actions', [])
                results = [{'role': 'tool', 'tool_call_id': action['call_id'], 'content': json.dumps(output)}
                           for action, output in zip(actions, outputs)]
                if not actions:
                    results.append({'role': 'user', 'content': json.dumps(host.idle())})
                if host.accepted is not None or host.stopped is not None:
                    results.append({'role': 'exit', 'content': 'compiler workflow finished',
                                    'extra': {'exit_status': 'validated' if host.accepted else 'stopped'}})
                return results

            def get_template_vars(self):
                return {}

            def serialize(self):
                return {}

        context = host.context()
        agent = DefaultAgent(Model(), Environment(), system_template='{{ instructions }}',
                             instance_template='{{ task }}', step_limit=0, cost_limit=0,
                             wall_time_limit_seconds=0, max_consecutive_format_errors=0, output_path=None)
        try:
            agent.run(task=json.dumps({k: v for k, v in context.items() if k != 'instructions'}),
                      instructions=context['instructions'])
            return {'agent_api_calls': transport.calls, 'agent_controller': 'mini-swe-agent'}
        finally:
            transport.client.close()
