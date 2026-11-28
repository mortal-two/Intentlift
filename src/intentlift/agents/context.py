"""Prepare the same actual CUDA request for the CLI and standalone MCP entry."""
import json
import os
from pathlib import Path
import subprocess
from ..core.errors import LiftError


def prepare_request(bench, clang, cuda_path):
    described = bench.describe()
    if not described.get('ok'):
        raise LiftError(described.get('kind', 'bench_failure'), described.get('message', 'Bench contract unavailable'))
    env = {k: v for k, v in os.environ.items() if not any(token in k.upper() for token in ('KEY', 'TOKEN', 'SECRET'))}
    env['PYTHONPATH'] = str(bench.root / 'src')
    probe = subprocess.run([bench.python, '-c', 'import torch,json; print(json.dumps(list(torch.cuda.get_device_capability())))'],
                           capture_output=True, text=True, env=env)
    if probe.returncode:
        raise LiftError('environment_failure', 'GPU capability query failed')
    major, minor = json.loads(probe.stdout)
    return {**described['contract'], 'clang': str(Path(clang).resolve()), 'cuda_path': cuda_path,
            'dsl_backend': bench.dsl_backend,
            'arch': f'sm_{major}{minor}'}
