"""Claude Code native authentication, model selection and MCP invocation."""
import json
from pathlib import Path
from .cli_runtime import CliBackend, copy_auth, write_config


class ClaudeBackend(CliBackend):
    name = 'claude'

    def prepare(self, host, home, server, instructions):
        copy_auth(Path.home() / '.claude/.credentials.json', home / '.claude/.credentials.json')
        source = Path.home() / '.claude.json'
        if source.is_file():
            data = json.loads(source.read_text())
            write_config(home / '.claude.json', {k: data[k] for k in ('oauthAccount', 'hasCompletedOnboarding') if k in data})
        config = home / 'mcp.json'
        write_config(config, {'mcpServers': {'intentlift': server}})
        args = ['--print', '--output-format', 'stream-json', '--verbose', '--no-session-persistence',
                '--strict-mcp-config', '--mcp-config', str(config), '--tools', '',
                '--allowedTools', 'mcp__intentlift__*', '--permission-mode', 'dontAsk',
                '--permission-prompts', 'none', '--disable-slash-commands', '--setting-sources', '',
                '--settings', '{"disableAllHooks":true}', '--append-system-prompt', instructions]
        if self.model:
            args += ['--model', self.model]
        return args
