"""External Agent invocation via the same MCP tools; no model authentication."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from .host import AgentHost, TOOLS, REPAIR_KINDS
from .context import prepare_request
from ..core.bench import Bench
from ..core.errors import LiftError
from ..core.sandbox import ROOT


def dispatch_for(host):
    def dispatch(request):
        if request['method'] == 'list':
            return TOOLS
        if request['method'] == 'context':
            return host.context()
        return host.invoke(request['name'], request.get('arguments', {}))
    return dispatch


class McpBackend:
    def __init__(self, model=None):
        pass

    def run(self, host, failure):
        from .mcp_bridge import serve
        serve(dispatch_for(host))
        return {'agent_api_calls': None, 'agent_controller': 'external-mcp-client'}


def standalone():
    parser = argparse.ArgumentParser(description='Intentlift controlled compiler tools over MCP stdio')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--tilelang-root', required=True)
    parser.add_argument('--bench', default=str(ROOT.parent / 'lift-bench'))
    parser.add_argument('--bench-python', default=sys.executable)
    parser.add_argument('--clang', default=shutil.which('clang++'))
    parser.add_argument('--cuda-path', default='/usr/local/cuda-12.8')
    parser.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix='intentlift-mcp-') as temporary:
            bench = Bench(args.bench, args.bench_python, args.dataset, args.case, args.tilelang_root, args.nvcc)
            host = AgentHost(prepare_request(bench, args.clang, args.cuda_path), bench, Path(temporary), backend='mcp')
            initial = host.compile()
            if not initial.get('ok') and initial.get('kind') not in REPAIR_KINDS:
                raise LiftError(initial['kind'], initial.get('message', 'compiler unavailable'))
            host.needs_repair = not initial.get('ok')
            host.failure = {k: v for k, v in initial.items() if k not in {'dsl', 'dsl_path'}}
            from .mcp_bridge import serve
            serve(dispatch_for(host))
            result = host.accepted or host.stopped or {'ok': False, 'kind': 'client_disconnected'}
            summary = {'ok': result['ok'], 'kind': result.get('kind'), 'agent_tool_calls': host.tool_calls}
        summary['temporary_artifacts_removed'] = not Path(temporary).exists()
        print(json.dumps(summary), file=sys.stderr)
        return 0
    except LiftError as exc:
        print(json.dumps(exc.record()), file=sys.stderr)
        return 1
