"""Capture checked, full-block bf16 addition and multiplication of direct members."""
import math
from intentlift.core.errors import require
from ._cuda_ast import children, index_values, qual_type, unwrap
from .cuda import member_axes


def peel(node):
    while node['kind'] in {'MaterializeTemporaryExpr', 'CXXBindTemporaryExpr', 'ExprWithCleanups'}:
        require(len(children(node)) == 1, 'unexpected bf16 temporary')
        node = children(node)[0]
    return unwrap(node)


def operator_call(node, name):
    require(node['kind'] == 'CXXOperatorCallExpr' and len(children(node)) == 3,
            'expected bf16 overloaded operator', node_kind=node['kind'])
    fn, left, right = children(node)
    require(children(fn) and children(fn)[0].get('referencedDecl', {}).get('name') == name,
            'unexpected bf16 operator')
    return left, right


def capture(parsed, context, registry):
    if any(k.startswith('staged_') for k in parsed):
        return parsed
    fn, contract = parsed['function'], parsed['contract']
    abi = contract['parameters']
    if not abi or any(p['dtype'] != 'bf16' for p in abi):
        return parsed
    outputs = [i for i, p in enumerate(abi) if p['mode'] != 'input']
    require(len(outputs) == 1 and len(abi) >= 3, 'bf16 arithmetic requires one output and two inputs')
    params = [p for p in children(fn) if p['kind'] == 'ParmVarDecl']
    require(len(params) == len(abi), 'CUDA parameter count mismatch', 'input_invalid')
    bindings = {}
    for slot, (decl, info) in enumerate(zip(params, abi)):
        ty = qual_type(decl).replace('__restrict', '').strip()
        require(ty.removeprefix('const ') in {'bfloat16_t *', 'cutlass::bfloat16_t *'}
                and info['name'] == decl['name'] and info['mode'] in {'input', 'output'}
                and (info['mode'] == 'input' or not ty.startswith('const '))
                and 1 <= len(info['shape']) <= 2
                and all(type(d) is int and d > 0 for d in info['shape']),
                'bf16 capture requires compatible parameter pointers', 'input_invalid')
        bindings[decl['id']] = {**info, 'slot': slot}
    shape = tuple(abi[outputs[0]]['shape'])
    n = math.prod(shape)
    require(0 < n <= 1024 and contract['grid'] == [1, 1, 1]
            and contract['block'] == [n, 1, 1] and contract['shared_mem'] == 0,
            'bf16 arithmetic requires one complete block')
    body = next(p for p in children(fn) if p['kind'] == 'CompoundStmt')
    require(len(children(body)) == 1, 'bf16 arithmetic requires one total store')
    statement = peel(children(body)[0])
    if statement['kind'] == 'BinaryOperator' and statement.get('opcode') == '=':
        target, rhs = children(statement)
    else:
        target, rhs = operator_call(statement, 'operator=')

    def access(node, mode):
        node = peel(node)
        require(node['kind'] == 'ArraySubscriptExpr', 'bf16 member must be directly indexed', node_kind=node['kind'])
        base, idx = children(node)
        base = unwrap(base)
        require(base['kind'] == 'DeclRefExpr' and base['referencedDecl']['id'] in bindings,
                'bf16 member must access an ABI parameter')
        info = bindings[base['referencedDecl']['id']]
        require(info['mode'] == mode, 'bf16 member access effect mismatch')
        return info, index_values(idx, n)

    dest, targets = access(target, 'output')
    require(dest['slot'] == outputs[0] and sorted(targets) == list(range(n)),
            'bf16 output must cover its members exactly once')
    rhs = peel(rhs)
    if rhs['kind'] == 'BinaryOperator':
        opcode = rhs.get('opcode')
        require(opcode in {'+', '*'} and len(children(rhs)) == 2,
                'bf16 expression must be addition or multiplication')
        operands = children(rhs)
    else:
        fn_ref = children(rhs)[0] if rhs['kind'] == 'CXXOperatorCallExpr' and len(children(rhs)) == 3 else {}
        name = (children(fn_ref)[0].get('referencedDecl', {}).get('name')
                if children(fn_ref) else None)
        require(name in {'operator+', 'operator*'}, 'unexpected bf16 operator')
        opcode = name[-1]
        operands = operator_call(rhs, name)
    require(opcode in registry.binary and registry.binary[opcode]['spelling'] == opcode,
            'bf16 arithmetic operator is not registered')
    reads = []
    for operand in operands:
        info, indices = access(operand, 'input')
        axes = member_axes(indices, tuple(info['shape']), targets, shape, allow_transpose=True)
        reads.append({'op': 'read', 'slot': info['slot'], 'axes': axes})
    result = {'parameters': abi, 'shape': shape, 'store': dest['slot'],
              'expression': {'op': registry.binary[opcode]['intent'], 'spelling': opcode, 'args': reads}}
    return {**parsed, 'staged_bf16': result}


def register(registry):
    registry.stage('cuda.bf16_pointwise', 'cuda_ast', 'cuda_ast', capture)
