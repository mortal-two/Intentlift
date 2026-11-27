"""Single-threaded local tool broker. CLI Agents never receive a write shell."""
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import tempfile
from .host import TOOLS
from ..core.errors import LiftError, require


LIMIT = 8 * 1024 * 1024


class ToolBroker:
    def __init__(self, host):
        self.host = host
        self.temporary = tempfile.TemporaryDirectory(prefix='lift-tools-')
        self.path = Path(self.temporary.name) / 'tools.sock'
        self.done = threading.Event()
        self.fatal = None
        broker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    raw = self.rfile.readline(LIMIT + 1)
                    require(len(raw) <= LIMIT, 'tool request exceeds limit', 'agent_protocol_failure')
                    request = json.loads(raw)
                    result = broker.dispatch(request)
                    response = {'ok': True, 'result': result}
                except LiftError as exc:
                    broker.fatal = exc
                    response = {'ok': False, 'error': exc.record()}
                except Exception as exc:
                    broker.fatal = LiftError('agent_protocol_failure', 'invalid local tool request: ' + type(exc).__name__)
                    response = {'ok': False, 'error': broker.fatal.record()}
                self.wfile.write(json.dumps(response, allow_nan=False).encode() + b'\n')
                self.wfile.flush()
                if broker.fatal or host.accepted is not None or host.stopped is not None:
                    broker.done.set()

        self.server = socketserver.UnixStreamServer(str(self.path), Handler)
        os.chmod(self.path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def dispatch(self, request):
        if request.get('method') == 'list':
            return TOOLS
        if request.get('method') == 'context':
            return self.host.context()
        require(request.get('method') == 'call', 'unknown broker method', 'agent_protocol_failure')
        return self.host.invoke(request['name'], request.get('arguments', {}))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.path.unlink(missing_ok=True)
        self.temporary.cleanup()


def rpc(path, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(json.dumps(request).encode() + b'\n')
        with client.makefile('rb') as stream:
            raw = stream.readline(LIMIT + 1)
    require(len(raw) <= LIMIT, 'tool response exceeds limit', 'agent_protocol_failure')
    response = json.loads(raw)
    if not response.get('ok'):
        error = response.get('error', {})
        raise LiftError(error.get('kind', 'agent_protocol_failure'), error.get('message', 'tool host rejected request'))
    return response['result']
