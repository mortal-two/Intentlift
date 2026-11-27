"""Codex CLI using only the protected compiler's MCP tools."""
import json
import os
from pathlib import Path
from .cli_runtime import CliBackend, copy_auth


class CodexBackend(CliBackend):
    name = 'codex'

    def prepare(self, host, home, server, instructions):
        (home / '.codex').mkdir(mode=0o700)
        copy_auth(Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')) / 'auth.json', home / '.codex/auth.json')
        args = ['exec', '--json', '--ephemeral', '--ignore-user-config', '--ignore-rules',
                '--skip-git-repo-check', '--sandbox', 'read-only', '-c', 'approval_policy="never"',
                '--disable', 'shell_tool', '--disable', 'unified_exec', '--disable', 'multi_agent',
                '--disable', 'apps', '-c', 'web_search="disabled"']
        if self.model:
            args += ['--model', self.model]
        config = {'developer_instructions': instructions,
                  'mcp_servers.intentlift.command': server['command'],
                  'mcp_servers.intentlift.args': server['args'],
                  'mcp_servers.intentlift.startup_timeout_sec': 60,
                  'mcp_servers.intentlift.tool_timeout_sec': 3600}
        config.update({'mcp_servers.intentlift.env.' + key: value for key, value in server['env'].items()})
        if os.environ.get('OPENAI_API_KEY'):
            config.update({'model_provider': 'intentlift', 'model_providers.intentlift.name': 'Intentlift',
                           'model_providers.intentlift.base_url': os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                           'model_providers.intentlift.env_key': 'OPENAI_API_KEY',
                           'model_providers.intentlift.wire_api': 'responses',
                           'model_providers.intentlift.request_max_retries': 0,
                           'model_providers.intentlift.stream_max_retries': 0,
                           'model_providers.intentlift.requires_openai_auth': False})
        for key, value in config.items():
            args += ['-c', key + '=' + json.dumps(value, ensure_ascii=False)]
        return [*args, '-']
