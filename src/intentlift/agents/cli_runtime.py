"""Sandboxed coding-CLI invocation over the shared local MCP broker."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .broker import ToolBroker
from .config import binary, authenticated
from ..core.errors import LiftError, require
from ..core.sandbox import ROOT


def python_path():
    # CLI homes are isolated; retain installed Python dependencies explicitly.
    return os.pathsep.join(dict.fromkeys([str(ROOT / 'src'),
        *[path for path in sys.path if path and 'site-packages' in Path(path).parts]]))


class CliBackend:
    name = None

    def __init__(self, model):
        self.model = model

    def prepare(self, host, home, server, instructions):
        raise NotImplementedError

    def environment(self, home):
        allowed = ('PATH', 'LANG', 'LC_ALL', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
                   'http_proxy', 'https_proxy', 'no_proxy', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE',
                   'NODE_EXTRA_CA_CERTS', 'LD_LIBRARY_PATH')
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        prefixes = {'codex': ('OPENAI_',), 'claude': ('ANTHROPIC_', 'CLAUDE_CODE_'),
                    'gemini': ('GEMINI_', 'GOOGLE_', 'GCLOUD_')}[self.name]
        for name, value in os.environ.items():
            if name.startswith(prefixes):
                env[name] = value
        env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
                   XDG_CACHE_HOME=str(home / '.cache'), XDG_STATE_HOME=str(home / '.state'),
                   PYTHONPATH=python_path(), PYTHONDONTWRITEBYTECODE='1',
                   TMPDIR='/tmp', CI='1', NO_COLOR='1')
        env['CODEX_HOME'] = str(home / '.codex')
        return env

    def command(self, host, broker):
        executable = binary(self.name)
        require(executable, f'{self.name} executable not found; set LIFT_{self.name.upper()}_BIN', 'configuration_failure')
        require(shutil.which('bwrap'), 'bubblewrap required for coding CLI backends', 'configuration_failure')
        home = host.work.resolve() / (self.name + '-home')
        home.mkdir(mode=0o700)
        server = {'command': sys.executable, 'args': ['-m', 'intentlift.agents.mcp_bridge'],
                  'env': {'LIFT_AGENT_SOCKET': str(broker.path), 'PYTHONPATH': python_path(),
                          'PYTHONDONTWRITEBYTECODE': '1'}}
        instructions = host.context()['instructions'] + '\nUse only the intentlift MCP tools. Native edits and shell commands are not the compiler modification path.'
        arguments = self.prepare(host, home, server, instructions)
        env = self.environment(home)
        command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
                   '--tmpfs', str(Path('/tmp').resolve()), '--unshare-pid', '--cap-drop', 'ALL',
                   '--die-with-parent', '--new-session', '--clearenv']
        for name in ('Intent', 'compilerinfra', 'CompilerInfra-lift'):
            path = ROOT.parent / name
            if path.is_dir():
                command += ['--tmpfs', str(path)]
        if host.bench.root.is_dir():
            command += ['--tmpfs', str(host.bench.root)]
        for path in dict.fromkeys((host.work.resolve(), broker.path.parent.resolve())):
            command += ['--ro-bind', str(path), str(path)]
        command += ['--bind', str(home), str(home), '--chdir', str(home)]
        for name, value in env.items():
            command += ['--setenv', name, value]
        return [*command, executable, *arguments]

    def run(self, host, failure):
        require(authenticated(self.name), f'{self.name} authentication is not configured; see agents configuration', 'configuration_failure')
        with ToolBroker(host) as broker:
            command = self.command(host, broker)
            payload = json.dumps(host.context(), ensure_ascii=False)
            # Files are confined to the temporary CLI home and removed with it.
            with (host.work / 'cli.stdout').open('w+') as stdout, (host.work / 'cli.stderr').open('w+') as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           text=True, start_new_session=True)
                try:
                    process.stdin.write(payload)
                    process.stdin.close()
                    while process.poll() is None and not broker.done.wait(0.2):
                        pass
                    if broker.done.is_set() and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        returncode = process.wait()
                    if broker.fatal:
                        raise broker.fatal
                    if host.accepted is None and host.stopped is None:
                        raise LiftError('agent_process_failure', f'{self.name} exited before validated completion',
                                        backend=self.name, returncode=returncode)
                    return {'agent_api_calls': None, 'agent_controller': self.name,
                            'agent_process_finish': 'validated' if host.accepted else 'stopped',
                            'agent_process_returncode': returncode}
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def copy_auth(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)


def write_config(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2))
    path.chmod(0o600)
