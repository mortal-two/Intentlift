"""Shared compiler Agent host. Every backend uses the same guarded tools."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from ..core.errors import LiftError, require
from ..core.sandbox import ROOT, EXTENSIONS, compiler_worker


STOP_KINDS = {'no_progress', 'boundary_required', 'dsl_expression_obstacle'}
REPAIR_KINDS = {'capability_missing', 'extension_invalid', 'generated_program_invalid', 'result_mismatch'}


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def semantic_state(root):
    """Ignore comments, layout and docstrings for no-progress detection."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).glob('*')):
        if not path.is_file() or path.suffix not in {'.py', '.json'}:
            continue
        digest.update(path.name.encode())
        try:
            if path.suffix == '.py':
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            node.body = node.body[1:]
                content = ast.dump(tree, include_attributes=False)
            else:
                content = json.dumps(json.loads(path.read_text()), sort_keys=True)
        except (SyntaxError, ValueError):
            content = path.read_text()
        digest.update(content.encode())
    return digest.hexdigest()


def function(name, description, properties=None, required=None):
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties or {},
                           'required': required or [], 'additionalProperties': False}}


TEXT = {'type': 'string'}
TOOLS = [
    function('task_context', 'Read the protected instructions, actual CUDA input and current failure.'),
    function('inspect', 'Query candidate component registrations and normal pipeline.'),
    function('read_file', 'Read compiler/knowledge by repository-relative path, component filename or approved absolute path.', {'path': TEXT}, ['path']),
    function('propose_changes', 'Write complete candidate extension files. Explain functional generality and preservation.',
             {'files': {'type': 'array', 'items': {'type': 'object',
                        'properties': {'path': TEXT, 'content': TEXT}, 'required': ['path', 'content'], 'additionalProperties': False}},
              'functional_basis': TEXT, 'preservation_analysis': TEXT}, ['files', 'functional_basis', 'preservation_analysis']),
    function('compile', 'Run normal compiler pipeline on the candidate extensions.'),
    function('validate', 'Recompile current candidate and run independent Bench on the exact generated DSL.'),
    function('stop', 'Report evidence for no progress, forbidden boundary, or proven DSL expression obstacle.',
             {'kind': {'type': 'string', 'enum': sorted(STOP_KINDS)}, 'evidence': TEXT,
              'next_requirement': TEXT}, ['kind', 'evidence', 'next_requirement']),
]


class AgentHost:
    def __init__(self, request, bench, work, model=None, backend="api", extension_root=None):
        self.request = request
        self.bench = bench
        self.work = Path(work).resolve()
        self.model = model
        self.backend = backend
        self._tool_lock = threading.RLock()
        self.extensions = Path(extension_root or EXTENSIONS).resolve(strict=True)
        protected = (ROOT / "src/intentlift/core", ROOT / "src/intentlift/agents", ROOT / "knowledge",
                     bench.root, ROOT.parent / "CompilerInfra-lift", ROOT.parent / "Intent", ROOT.parent / "compilerinfra")
        require(not any(self.extensions.is_relative_to(path) for path in protected),
                "invalid extension maintenance root", "boundary_required")
        self.needs_repair = True
        self.failure = {}
        self.tool_calls = 0
        self.seen_idle = set()
        self.failed_states = set()
        self.warned_states = set()
        self.candidate = self.work / 'candidate'
        shutil.copytree(self.extensions, self.candidate, ignore=shutil.ignore_patterns('__pycache__'))
        self.original = fingerprint(self.extensions)
        self.protected = {path: fingerprint(path) for path in (
            ROOT / 'src/intentlift/core', ROOT / 'src/intentlift/agents', ROOT / 'knowledge', bench.root)}
        self.attempt = 0
        self.accepted = None
        self.stopped = None
        self.changes = []
        self.observations = []

    def compile(self):
        self.attempt += 1
        result = compiler_worker({**self.request, 'extension_root': str(self.candidate)})
        record = {k: v for k, v in result.items() if k != 'dsl'}
        record.update({'compiler_sha256': fingerprint(self.candidate), 'action': 'compile'})
        self.observations.append(record)
        if result.get('ok'):
            path = self.work / f'generated_{self.attempt}.intent.py'
            path.write_text(result['dsl'])
            result['dsl_path'] = str(path)
        return result

    def validate(self):
        self.check_protected()
        result = self.compile()
        if not result.get('ok'):
            return result
        dsl_path = result['dsl_path']
        result = self.bench.validate(dsl_path, self.work / f'validation_{self.attempt}')
        if not result.get('ok') and result.get('kind') in REPAIR_KINDS:
            self.needs_repair = True
        self.check_protected()
        self.observations.append({'action': 'validate', 'compiler_sha256': fingerprint(self.candidate),
                                  'kind': result['kind'], 'ok': result['ok']})
        if result.get('ok'):
            require(self.changes or not self.needs_repair, 'Agent must explain a general compiler change before acceptance', 'boundary_required')
            require(fingerprint(self.extensions) == self.original, 'official compiler changed during candidate iteration', 'boundary_required')
            for path in self.candidate.iterdir():
                if path.is_file() and path.suffix in {'.py', '.json'}:
                    (self.extensions / path.name).write_bytes(path.read_bytes())
            self.accepted = {**result, 'compiler_sha256': fingerprint(self.extensions),
                             'dsl_path': dsl_path,
                             'accepted_components': sorted({f for change in self.changes for f in change['files']})}
        return result

    def check_protected(self):
        for path, digest in self.protected.items():
            require(fingerprint(path) == digest, 'protected framework, knowledge or Bench content changed during iteration',
                    'boundary_required')

    def read_file(self, path):
        requested = Path(path).expanduser()
        if requested.is_absolute():
            target = requested.resolve()
            if target == Path(self.request['source']).resolve():
                return {'ok': True, 'content': target.read_text()}
            for extension in (self.candidate, self.extensions, EXTENSIONS):
                if target.is_relative_to(extension):
                    target = (self.candidate / target.relative_to(extension)).resolve()
                    root = self.candidate
                    break
            else:
                if target.is_relative_to(ROOT / 'src/intentlift'):
                    root = ROOT / 'src/intentlift'
                elif target.is_relative_to(ROOT / 'knowledge'):
                    root = ROOT / 'knowledge'
                elif target in (ROOT / 'README.md', ROOT / 'AGENTS.md'):
                    return {'ok': True, 'content': target.read_text()}
                else:
                    raise LiftError('boundary_required', 'read path outside the compiler, knowledge and selected source', requested_path=path)
        else:
            parts = requested.parts
            if parts[:3] == ('src', 'intentlift', 'extensions'):
                root, suffix = self.candidate, Path(*parts[3:])
            elif parts[:1] == ('extensions',):
                root, suffix = self.candidate, Path(*parts[1:])
            elif len(parts) == 1 and (self.candidate / requested).is_file():
                root, suffix = self.candidate, requested
            elif parts[:2] == ('src', 'intentlift'):
                root, suffix = ROOT / 'src/intentlift', Path(*parts[2:])
            elif parts[:1] in {('core',), ('agents',)}:
                root, suffix = ROOT / 'src/intentlift', requested
            elif parts[:1] == ('knowledge',):
                root, suffix = ROOT / 'knowledge', Path(*parts[1:])
            elif str(requested) in {'README.md', 'AGENTS.md'}:
                return {'ok': True, 'content': (ROOT / requested).read_text()}
            else:
                raise LiftError('boundary_required', 'read access limited to compiler and knowledge', requested_path=path)
            target = (root / suffix).resolve()
        require(target.is_relative_to(root) and target.suffix in {'.py', '.json', '.md'},
                'invalid read path', 'boundary_required', requested_path=path)
        if not target.is_file():
            return {'ok': False, 'kind': 'extension_invalid', 'message': 'Compiler file not found; component filenames may be read directly.',
                    'requested_path': path, 'available_components': sorted(p.name for p in self.candidate.iterdir() if p.is_file())}
        return {'ok': True, 'content': target.read_text()}

    def propose_changes(self, files, functional_basis, preservation_analysis):
        require(self.needs_repair, "normal compilation needs no Agent modification", "boundary_required")
        require(functional_basis.strip() and preservation_analysis.strip(), 'generality and preservation explanations required', 'boundary_required')
        require(files, 'empty changes do not add a capability', 'boundary_required')
        staged = {}
        for item in files:
            relative = Path(item['path'])
            require(len(relative.parts) == 1 and relative.name not in {'__init__.py', '.', '..'}
                    and (relative.suffix == '.py' or relative.name == 'pipeline.json'),
                    'only direct extension Python files and pipeline.json are writable', 'boundary_required')
            content = item['content']
            require(len(content) <= 200_000, 'component too large', 'boundary_required')
            for identity in (self.request['entry'], self.request['source_sha256'], Path(self.request['source']).name):
                require(identity not in content, 'Kernel-specific identity forbidden in compiler rules', 'boundary_required')
            if relative.suffix == '.py':
                ast.parse(content)
            else:
                data = json.loads(content)
                require(isinstance(data, list) and data and all(isinstance(x, str) for x in data),
                        'invalid pipeline list', 'extension_invalid')
            staged[relative.name] = content
        before = fingerprint(self.candidate)
        for name, content in staged.items():
            (self.candidate / name).write_text(content)
        after = fingerprint(self.candidate)
        self.changes.append({'files': list(staged), 'functional_basis': functional_basis,
                             'preservation_analysis': preservation_analysis,
                             'changed': before != after})
        return {'ok': True, 'changed': before != after, 'compiler_sha256': after}

    def dispatch(self, name, args):
        if name == 'task_context':
            return {'ok': True, **self.context()}
        if name == 'inspect':
            return compiler_worker({'action': 'inspect', 'extension_root': str(self.candidate)})
        if name == 'read_file':
            return self.read_file(**args)
        if name == 'propose_changes':
            return self.propose_changes(**args)
        if name == 'compile':
            return self.compile()
        if name == 'validate':
            return self.validate()
        if name == 'stop':
            require(args['kind'] in STOP_KINDS and args['evidence'].strip() and args['next_requirement'].strip(),
                    'stop requires concrete evidence and next requirement', 'boundary_required')
            self.stopped = {'ok': False, **args}
            return self.stopped
        raise LiftError('boundary_required', 'unknown tool requested')

    def context(self):
        return {'instructions': (ROOT / 'knowledge/compiler.md').read_text(),
                'failure': self.failure, 'request': self.request,
                'cuda_source': Path(self.request['source']).read_text(),
                'needs_repair': self.needs_repair, 'observations': self.observations}

    def idle(self):
        state = semantic_state(self.candidate)
        if state in self.seen_idle:
            self.stopped = {'ok': False, 'kind': 'no_progress',
                            'evidence': 'unchanged compiler and repeated response without an actionable tool call',
                            'next_requirement': 'a distinct evidence-based compiler modification plan'}
        self.seen_idle.add(state)
        return {'ok': False, 'kind': 'extension_invalid',
                'message': 'Use the registered tools for an evidence-based next step or stop with concrete evidence.'}

    def invoke(self, name, args):
        # MCP clients may submit tools concurrently; candidate changes and Bench
        # acceptance must observe one ordered compiler state across all backends.
        with self._tool_lock:
            try:
                return self._invoke(name, args)
            except LiftError as exc:
                self.stopped = exc.record()
                raise

    def _invoke(self, name, args):
        from jsonschema import Draft202012Validator
        schemas = {tool['name']: tool['parameters'] for tool in TOOLS}
        require(name in schemas, 'unknown tool requested', 'boundary_required')
        errors = list(Draft202012Validator(schemas[name]).iter_errors(args))
        if errors:
            return {'ok': False, 'kind': 'extension_invalid', 'message': errors[0].message[:1000]}
        if (self.accepted is not None or self.stopped is not None) and name not in {'task_context', 'inspect', 'read_file'}:
            return self.accepted or self.stopped
        self.check_protected()
        self.tool_calls += 1
        try:
            result = self.dispatch(name, args)
        except LiftError:
            raise
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            result = {'ok': False, 'kind': 'extension_invalid', 'message': str(exc)[:1000]}
        if name in {'compile', 'validate'} and not result.get('ok') and result.get('kind') in REPAIR_KINDS:
            signature = (semantic_state(self.candidate), result.get('kind'), result.get('message'),
                         result.get('stage'), result.get('cuda_opcode'))
            if signature in self.failed_states:
                if signature in self.warned_states:
                    self.stopped = {'ok': False, 'kind': 'no_progress',
                                    'evidence': 'identical compiler state and failure repeated after explicit no-progress feedback',
                                    'next_requirement': 'a distinct evidence-based compiler modification'}
                else:
                    result['next_step_required'] = 'Identical failure with unchanged code. Modify with new evidence or report no_progress.'
                    self.warned_states.add(signature)
            self.failed_states.add(signature)
        if not result.get('ok') and result.get('kind') not in REPAIR_KINDS | STOP_KINDS:
            raise LiftError(result.get('kind', 'infrastructure_failure'), result.get('message', 'execution infrastructure unavailable'))
        return result

    def run(self, failure):
        from .config import create_backend
        self.failure = failure
        self.needs_repair = True
        backend = create_backend(self.backend, self.model)
        statistics = backend.run(self, failure)
        require(self.accepted is not None or self.stopped is not None,
                'Agent exited before validation or an evidenced stop', 'agent_protocol_failure')
        result = {**(self.accepted or self.stopped), 'agent_backend': self.backend,
                  'agent_tool_calls': self.tool_calls, **statistics}
        safe = {'initial_failure': failure, 'result': result,
                'compiler_changes': self.changes, 'observations': self.observations}
        (self.work / 'agent_result.json').write_text(json.dumps(safe, indent=2))
        return result

