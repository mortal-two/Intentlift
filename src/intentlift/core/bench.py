"""Call the independent Bench executable; never import Bench implementation."""
import json
import os
from pathlib import Path
import subprocess
from .errors import LiftError


class Bench:
    def __init__(self, root, python, dataset, case, tilelang_root, nvcc):
        self.root = Path(root).resolve()
        self.python = python
        self.common = ['--dataset', dataset, '--case', case, '--tilelang-root', tilelang_root]
        self.nvcc = nvcc

    def invoke(self, action, *args):
        env = {k: v for k, v in os.environ.items() if 'KEY' not in k and 'TOKEN' not in k
               and k not in {'OPENAI_BASE_URL', 'OPENAI_MODEL'}}
        env['PYTHONPATH'] = str(self.root / 'src')
        command = [self.python, '-m', 'lift_bench.cli', action, *self.common, *map(str, args)]
        completed = subprocess.run(command, text=True, capture_output=True, env=env)
        try:
            result = json.loads(completed.stdout)
        except ValueError as exc:
            raise LiftError('bench_failure', 'Bench did not return valid JSON', diagnostic=completed.stderr[-2000:]) from exc
        if completed.returncode not in (0, 1) or bool(result.get('ok')) != (completed.returncode == 0):
            raise LiftError('bench_failure', 'Bench process status does not agree with result')
        return result

    def describe(self):
        return self.invoke('describe')

    def validate(self, dsl, work):
        return self.invoke('validate', '--dsl', dsl, '--work', work, '--nvcc', self.nvcc)
