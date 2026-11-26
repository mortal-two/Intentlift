"""API function-calling host with controlled candidate writes and no shell tool."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
from .errors import LiftError, require
from .sandbox import ROOT, EXTENSIONS, compiler_worker


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
    function('inspect', 'Query candidate component registrations and normal pipeline.'),
    function('read_file', 'Read compiler source or fixed knowledge. Path relative to Intentlift root.', {'path': TEXT}, ['path']),
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
    def __init__(self, request, bench, work, model):
        self.request = request
        self.bench = bench
        self.work = Path(work)
        self.model = model
        self.candidate = self.work / 'candidate'
        shutil.copytree(EXTENSIONS, self.candidate, ignore=shutil.ignore_patterns('__pycache__'))
        self.original = fingerprint(EXTENSIONS)
        self.protected = {path: fingerprint(path) for path in (
            ROOT / 'src/intentlift/core', ROOT / 'knowledge', bench.root)}
        self.attempt = 0
        self.accepted = None
        self.stopped = None
        self.changes = []
        self.observations = []
        self.change_since_validation = False

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
        self.check_protected()
        self.observations.append({'action': 'validate', 'compiler_sha256': fingerprint(self.candidate),
                                  'kind': result['kind'], 'ok': result['ok']})
        if result.get('ok'):
            require(self.changes, 'Agent must explain a general compiler change before acceptance', 'boundary_required')
            require(fingerprint(EXTENSIONS) == self.original, 'official compiler changed during candidate iteration', 'boundary_required')
            for path in self.candidate.iterdir():
                if path.is_file() and path.suffix in {'.py', '.json'}:
                    (EXTENSIONS / path.name).write_bytes(path.read_bytes())
            self.accepted = {**result, 'compiler_sha256': fingerprint(EXTENSIONS),
                             'dsl_path': dsl_path,
                             'accepted_components': sorted({f for change in self.changes for f in change['files']})}
        return result

    def check_protected(self):
        for path, digest in self.protected.items():
            require(fingerprint(path) == digest, 'protected framework, knowledge or Bench content changed during iteration',
                    'boundary_required')

    def read_file(self, path):
        relative = Path(path)
        if relative.parts[:3] == ('src', 'intentlift', 'extensions'):
            root = self.candidate
            suffix = Path(*relative.parts[3:])
        elif relative.parts[:2] == ('src', 'intentlift') or relative.parts[:1] == ('knowledge',):
            root = ROOT
            suffix = relative
        else:
            raise LiftError('boundary_required', 'read access limited to compiler and knowledge')
        target = (root / suffix).resolve()
        require(target.is_relative_to(root) and target.is_file() and target.suffix in {'.py', '.json', '.md'},
                'invalid read path', 'boundary_required')
        return {'ok': True, 'content': target.read_text()}

    def propose_changes(self, files, functional_basis, preservation_analysis):
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

    def run(self, failure):
        from openai import OpenAI, OpenAIError
        require(self.model, 'set --model or OPENAI_MODEL for API Agent', 'configuration_failure')
        require(os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY required for API Agent', 'configuration_failure')
        client = OpenAI(max_retries=0, timeout=float(os.environ.get('LIFT_API_TIMEOUT', '600')))
        knowledge = (ROOT / 'knowledge/compiler.md').read_text()
        source = Path(self.request['source']).read_text()
        history = [{'role': 'user', 'content': json.dumps({'failure': failure, 'request': self.request,
                                                        'cuda_source': source, 'current_observations': self.observations})}]
        calls = 0
        seen_idle = set()
        failed_states = set()
        warned_states = set()
        while self.accepted is None and self.stopped is None:
            try:
                response = client.responses.create(model=self.model, instructions=knowledge, tools=TOOLS,
                                                   parallel_tool_calls=False, input=history)
            except OpenAIError as exc:
                raise LiftError('api_failure', 'API request failed: ' + type(exc).__name__, api_calls=calls + 1) from None
            calls += 1
            require(response.status == 'completed', 'API response incomplete', 'api_failure')
            history.extend(item.model_dump(exclude_none=True) for item in response.output)
            tool_calls = [item for item in response.output if item.type == 'function_call']
            if not tool_calls:
                state = semantic_state(self.candidate)
                if state in seen_idle:
                    self.stopped = {'ok': False, 'kind': 'no_progress',
                                    'evidence': 'same compiler state and repeated non-actionable API response',
                                    'next_requirement': 'a distinct evidence-based modification plan'}
                    break
                seen_idle.add(state)
                history.append({'role': 'user', 'content': 'No validation has passed. Use the tools for an evidence-based next step, or stop with concrete evidence.'})
                continue
            for call in tool_calls:
                try:
                    args = json.loads(call.arguments)
                    result = self.dispatch(call.name, args)
                except LiftError:
                    raise
                except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                    result = {'ok': False, 'kind': 'extension_invalid', 'message': str(exc)[:1000]}
                # An unsuccessful execution with unchanged code and identical
                # failure cannot become progress by calling it again. Give the
                # model the concrete repetition before accepting another retry.
                if call.name in {'compile', 'validate'} and not result.get('ok') and result.get('kind') in REPAIR_KINDS:
                    signature = (semantic_state(self.candidate), result.get('kind'),
                                 result.get('message'), result.get('stage'), result.get('cuda_opcode'))
                    if signature in failed_states:
                        if signature in warned_states:
                            self.stopped = {'ok': False, 'kind': 'no_progress',
                                            'evidence': 'unchanged compiler and identical failed action repeated after explicit no-progress feedback',
                                            'next_requirement': 'a distinct evidence-based compiler modification'}
                        else:
                            result['next_step_required'] = 'Identical failure at unchanged code. Make a distinct evidence-based modification or report no_progress; repeating this action has no new evidence.'
                            warned_states.add(signature)
                    failed_states.add(signature)
                history.append({'type': 'function_call_output', 'call_id': call.call_id,
                                'output': json.dumps(result, allow_nan=False)})
                if not result.get('ok') and result.get('kind') not in REPAIR_KINDS | STOP_KINDS:
                    raise LiftError(result.get('kind', 'infrastructure_failure'), result.get('message', 'execution infrastructure unavailable'))
                if self.accepted or self.stopped:
                    break
        result = self.accepted or self.stopped
        result = {**result, 'agent_api_calls': calls}
        safe = {'initial_failure': failure, 'result': result,
                'compiler_changes': self.changes, 'observations': self.observations}
        (self.work / 'agent_result.json').write_text(json.dumps(safe, indent=2))
        return result
