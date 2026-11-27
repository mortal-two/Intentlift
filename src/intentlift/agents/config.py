"""Backend catalog, lazy construction, public credential and binary settings."""
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
from ..core.errors import LiftError


BACKENDS = {
    'api': ('responses', 'ResponsesBackend', 'OPENAI_MODEL'),
    'responses': ('responses', 'ResponsesBackend', 'OPENAI_MODEL'),
    'chat-completions': ('chat_completions', 'ChatCompletionsBackend', 'OPENAI_MODEL'),
    'codex': ('codex', 'CodexBackend', 'OPENAI_MODEL'),
    'miniswe': ('miniswe', 'MiniSweBackend', 'OPENAI_MODEL'),
    'mcp': ('mcp', 'McpBackend', None),
    'claude': ('claude', 'ClaudeBackend', 'ANTHROPIC_MODEL'),
    'gemini': ('gemini', 'GeminiBackend', 'GEMINI_MODEL'),
}


def binary(name):
    override = os.environ.get('LIFT_' + name.upper() + '_BIN')
    if override:
        return shutil.which(override) or (override if Path(override).is_file() and os.access(override, os.X_OK) else None)
    found = shutil.which(name)
    if found:
        return found
    local = Path.home() / '.local/share/intentlift/agent-clis/node_modules/.bin' / name
    return str(local) if local.is_file() and os.access(local, os.X_OK) else None


def authenticated(name):
    if name in {'api', 'responses', 'chat-completions', 'miniswe'}:
        return bool(os.environ.get('OPENAI_API_KEY'))
    if name == 'codex':
        return bool(os.environ.get('OPENAI_API_KEY')) or (Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')) / 'auth.json').is_file()
    if name == 'claude':
        return any(os.environ.get(k) for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN')) or (Path.home() / '.claude/.credentials.json').is_file()
    if name == 'gemini':
        return (any(os.environ.get(k) for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS'))
                or os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true'
                or (Path.home() / '.gemini/oauth_creds.json').is_file())
    return True


def create_backend(name, model=None):
    if name not in BACKENDS:
        raise LiftError('configuration_failure', 'unknown Agent backend: ' + name)
    module, cls, variable = BACKENDS[name]
    selected = model or (os.environ.get(variable) if variable else None)
    return getattr(importlib.import_module('.' + module, __package__), cls)(selected)


def catalog():
    entries = []
    for name, (module, cls, variable) in BACKENDS.items():
        if name == 'responses':
            continue
        executable = binary(name) if name in {'codex', 'claude', 'gemini'} else None
        requirements = ['mcp'] if name in {'mcp', 'codex', 'claude', 'gemini'} else ['minisweagent'] if name == 'miniswe' else ['openai']
        missing = [item for item in requirements if importlib.util.find_spec(item) is None]
        if name in {'codex', 'claude', 'gemini'} and not executable:
            missing.append(name + ' executable')
        entries.append({'backend': name, 'module': 'intentlift.agents.' + module,
                        'default': name == 'api', 'model_variable': variable,
                        'model': os.environ.get(variable) if variable else None,
                        'executable': executable, 'dependencies_ready': not missing,
                        'missing': missing, 'authentication_configured': authenticated(name)})
    return {'ok': True, 'backends': entries}
