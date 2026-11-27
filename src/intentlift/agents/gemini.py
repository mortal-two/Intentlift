"""Gemini CLI native API/OAuth/Vertex configuration with restricted MCP tools."""
import os
from pathlib import Path
from .cli_runtime import CliBackend, copy_auth, write_config
from .host import TOOLS


class GeminiBackend(CliBackend):
    name = 'gemini'

    def environment(self, home):
        env = super().environment(home)
        env['GEMINI_CLI_HOME'] = str(home)
        # The Gemini API auth mode reads GEMINI_API_KEY; GOOGLE_API_KEY is
        # otherwise consumed only by Vertex in the native CLI.
        if not env.get('GEMINI_API_KEY') and env.get('GOOGLE_API_KEY'):
            env['GEMINI_API_KEY'] = env['GOOGLE_API_KEY']
        adc = home / '.config/gcloud/application_default_credentials.json'
        if adc.is_file():
            env['GOOGLE_APPLICATION_CREDENTIALS'] = str(adc)
        return env

    def prepare(self, host, home, server, instructions):
        copy_auth(Path.home() / '.gemini/oauth_creds.json', home / '.gemini/oauth_creds.json')
        if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true' or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            auth = 'vertex-ai'
            adc = Path(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or
                       (Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) /
                        'gcloud/application_default_credentials.json'))
            copy_auth(adc, home / '.config/gcloud/application_default_credentials.json')
        elif os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
            auth = 'gemini-api-key'
        else:
            auth = 'oauth-personal'
        write_config(home / '.gemini/settings.json', {
            'security': {'auth': {'selectedType': auth}, 'folderTrust': {'enabled': False}},
            'mcpServers': {'intentlift': {**server, 'trust': True, 'includeTools': [t['name'] for t in TOOLS]}},
            'tools': {'core': []},
            'model': {'maxSessionTurns': -1},
            'general': {'sessionRetention': {'enabled': False}},
        })
        (home / 'GEMINI.md').write_text(instructions)
        policy = home / 'intentlift-policy.toml'
        policy.write_text('[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 900\n\n'
                          '[[rule]]\ntoolName = "*"\nmcpName = "intentlift"\ndecision = "allow"\npriority = 901\n')
        args = ['--prompt', 'Use the supplied compiler task and only intentlift MCP tools.',
                '--output-format', 'stream-json', '--approval-mode', 'yolo', '--skip-trust',
                '--allowed-mcp-server-names', 'intentlift', '--policy', str(policy)]
        if self.model:
            args += ['--model', self.model]
        return args
