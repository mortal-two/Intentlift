"""Backend-neutral model turns; no backend executes unrestricted model tools."""
from dataclasses import dataclass
import json
from ..core.errors import require


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Turn:
    text: str
    calls: tuple


def arguments(raw):
    value = json.loads(raw) if isinstance(raw, str) else raw
    require(isinstance(value, dict), 'tool arguments must be a JSON object', 'agent_protocol_failure')
    return value


def tool_loop(host, transport):
    context = host.context()
    transport.start(context['instructions'], {k: v for k, v in context.items() if k != 'instructions'})
    while host.accepted is None and host.stopped is None:
        turn = transport.next_turn()
        if not turn.calls:
            transport.feedback(host.idle())
            continue
        outputs = []
        for call in turn.calls:
            outputs.append((call, host.invoke(call.name, call.arguments)))
            if host.accepted is not None or host.stopped is not None:
                break
        transport.observe(outputs)
    return {'agent_api_calls': transport.calls}
