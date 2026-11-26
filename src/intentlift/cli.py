import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from .core.agent import AgentHost, REPAIR_KINDS, fingerprint
from .core.bench import Bench
from .core.errors import LiftError
from .core.sandbox import ROOT, EXTENSIONS, compiler_worker


def main():
    parser = argparse.ArgumentParser(description='Compile CUDA to IntentDSL; validate every completed run independently')
    parser.add_argument('action', choices=['run', 'inspect'])
    parser.add_argument('--dataset')
    parser.add_argument('--case')
    parser.add_argument('--tilelang-root')
    parser.add_argument('--bench', default=str(ROOT.parent / 'lift-bench'))
    parser.add_argument('--bench-python', default=sys.executable)
    parser.add_argument('--clang', default=shutil.which('clang++'))
    parser.add_argument('--cuda-path', default='/usr/local/cuda-12.8')
    parser.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL'))
    parser.add_argument('--run-root', help='Optional parent for temporary work; each run is removed on completion')
    parser.add_argument('--output', help='Explicit destination for the validated IntentDSL deliverable')
    parser.add_argument('--no-agent', action='store_true', help='Report missing capability without calling the API')
    args = parser.parse_args()
    work = None
    temporary = None
    dsl = None
    agent_invoked = False
    try:
        if args.action == 'inspect':
            result = compiler_worker({'action': 'inspect', 'extension_root': str(EXTENSIONS)})
        else:
            if not all((args.dataset, args.case, args.tilelang_root, args.clang)):
                raise LiftError('configuration_failure', '--dataset, --case, --tilelang-root and Clang required')
            run_root = Path(args.run_root or tempfile.gettempdir()).resolve()
            for repo in (ROOT, Path(args.bench).resolve(), ROOT.parent / 'CompilerInfra-lift'):
                if run_root.is_relative_to(repo):
                    raise LiftError('configuration_failure', 'run root must be outside repositories')
            run_root.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix='intentlift-', dir=run_root)
            work = Path(temporary.name)
            bench = Bench(args.bench, args.bench_python, args.dataset, args.case, args.tilelang_root, args.nvcc)
            described = bench.describe()
            if not described['ok']:
                result = described
            else:
                contract = described['contract']
                import subprocess
                env = dict(os.environ)
                env['PYTHONPATH'] = str(Path(args.bench).resolve() / 'src')
                probe = subprocess.run([args.bench_python, '-c',
                    'import torch,json; print(json.dumps(list(torch.cuda.get_device_capability())))'],
                    capture_output=True, text=True, env=env)
                if probe.returncode:
                    raise LiftError('environment_failure', 'GPU capability query failed')
                major, minor = json.loads(probe.stdout)
                request = {**contract, 'clang': str(Path(args.clang).resolve()), 'cuda_path': args.cuda_path,
                           'arch': f'sm_{major}{minor}', 'extension_root': str(EXTENSIONS)}
                result = compiler_worker(request)
                if result.get('ok'):
                    dsl = work / 'generated.intent.py'
                    dsl.write_text(result['dsl'])
                    result = bench.validate(dsl, work / 'validation')
                if not result.get('ok') and result.get('kind') in REPAIR_KINDS and not args.no_agent:
                    agent_invoked = True
                    result = AgentHost(request, bench, work, args.model).run(result)
                if result.get('ok'):
                    validated = Path(result.get('dsl_path') or dsl)
                    if args.output:
                        destination = Path(args.output).resolve()
                        protected = (ROOT, Path(args.bench).resolve(), ROOT.parent / 'CompilerInfra-lift',
                                     ROOT.parent / 'Intent', ROOT.parent / 'compilerinfra', work)
                        if any(destination.is_relative_to(root) for root in protected):
                            raise LiftError('configuration_failure', 'DSL deliverable must be outside code and temporary work directories')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_text(validated.read_text())
                        result['output'] = str(destination)
                    else:
                        result['dsl'] = validated.read_text()
                result = {**result, 'compiler_sha256': fingerprint(EXTENSIONS),
                          'agent_invoked': agent_invoked}
    except LiftError as exc:
        result = exc.record()
    except Exception as exc:
        result = {'ok': False, 'kind': 'infrastructure_failure', 'message': f'{type(exc).__name__}: {exc}'}
    finally:
        if temporary is not None:
            temporary.cleanup()
    result.setdefault('agent_invoked', agent_invoked)
    if work:
        for field in ('evidence', 'run_directory', 'dsl_path'):
            result.pop(field, None)
        result['temporary_artifacts_removed'] = not work.exists()
    print(json.dumps(result, allow_nan=False))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
