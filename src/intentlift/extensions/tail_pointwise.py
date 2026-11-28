"""Prove guarded multi-block f32 pointwise stores by enumerating active launch lanes."""
import math
from intentlift.core.errors import require
from . import cuda, signed_byte_pointwise
from ._cuda_ast import assignment, children, guard, index_values, qual_type, unwrap


def capture(parsed, context, registry):
    contract = parsed['contract']
    abi = contract['parameters']
    # A shared-memory scan is not pointwise. Route its six-statement staging
    # schedule to the existing scan verifier before applying rank restrictions.
    # The verifier checks the resolved CumSum2D call, barriers, shared and
    # global coordinates and ownership; this dispatch alone proves nothing.
    body = next(p for p in children(parsed['function']) if p['kind'] == 'CompoundStmt')
    statements = children(body)
    if (len(statements) == 6 and statements[0]['kind'] == 'DeclStmt'
            and len(children(statements[0])) == 1
            and qual_type(children(statements[0])[0]) == 'float[]'
            and statements[3]['kind'] == 'CallExpr'):
        return cuda.capture(parsed, context, registry)
    # Leave the existing single-block and non-f32 capture paths untouched.
    if (contract['grid'] == [1, 1, 1] or not abi
            or any(p['dtype'] != 'f32' for p in abi)):
        return signed_byte_pointwise.capture(parsed, context, registry)
    outputs = [slot for slot, p in enumerate(abi) if p['mode'] != 'input']
    require(len(outputs) == 1 and all(p['mode'] in {'input', 'output'} for p in abi),
            'guarded pointwise requires independent inputs and one output')
    shape = tuple(abi[outputs[0]]['shape'])
    require(len(shape) in {1, 2} and all(type(d) is int and d > 0 for d in shape),
            'guarded pointwise needs a positive static member domain')
    n = math.prod(shape)
    grid, block = contract['grid'], contract['block']
    require(0 < n <= 1048576 and all(type(x) is int and 0 < x <= 1024 for x in grid)
            and block[1:] == [1, 1] and type(block[0]) is int and 0 < block[0] <= 1024
            and math.prod(grid) * block[0] <= 1048576 and contract['shared_mem'] == 0,
            'guarded pointwise launch is outside checked enumeration bounds')
    fn = parsed['function']
    params = [p for p in children(fn) if p['kind'] == 'ParmVarDecl']
    require(len(params) == len(abi), 'CUDA ABI parameter count mismatch', 'input_invalid')
    bindings = {}
    for slot, (decl, info) in enumerate(zip(params, abi)):
        ty = qual_type(decl).replace('__restrict', '').strip()
        require(ty.removeprefix('const ') == 'float *' and info['name'] == decl['name']
                and (info['mode'] == 'input' or not ty.startswith('const '))
                and len(info['shape']) in {1, 2}
                and all(type(d) is int and d > 0 for d in info['shape']),
                'guarded pointwise requires compatible f32 parameter views')
        bindings[decl['id']] = {**info, 'slot': slot}
    require(len(statements) == 1 and statements[0]['kind'] == 'IfStmt',
            'guarded pointwise needs a sole side-effect-free guarded store')
    stores = []
    for bx in range(grid[0]):
        for by in range(grid[1]):
            for bz in range(grid[2]):
                block_coord = (bx, by, bz)
                active, guarded = guard(statements[0], block[0], block_coord)
                require(len(guarded) == 1, 'guarded pointwise needs one assignment')
                destination, value = assignment(guarded[0])
                stores.append((block_coord, active, destination, value))

    def access(node, mode):
        node = unwrap(node)
        require(node['kind'] == 'ArraySubscriptExpr', 'guarded member access must be a direct subscript')
        base, index = children(node)
        base = unwrap(base)
        require(base['kind'] == 'DeclRefExpr' and base['referencedDecl']['id'] in bindings,
                'guarded member must access an ABI parameter')
        info = bindings[base['referencedDecl']['id']]
        require(info['mode'] == mode, 'guarded member access effect mismatch')
        values = []
        for coord, active, _, _ in stores:
            indices = index_values(index, block[0], block=coord)
            values.extend(indices[i] for i in active)
        require(all(0 <= x < math.prod(info['shape']) for x in values),
                'active guarded member index exceeds its view')
        return info, tuple(values)

    first_dest, first_value = stores[0][2:]
    target, targets = access(first_dest, 'output')
    require(target['slot'] == outputs[0] and len(targets) == n
            and sorted(targets) == list(range(n)),
            'active guarded stores must cover output exactly once')
    require(all(dest is first_dest and value is first_value for _, _, dest, value in stores),
            'guarded stores must share one expression')

    def expression(node):
        node = unwrap(node)
        require(qual_type(node) in {'float', 'const float'}, 'guarded expression must be f32')
        if node['kind'] == 'ArraySubscriptExpr':
            info, indices = access(node, 'input')
            axes = cuda.member_axes(indices, tuple(info['shape']), targets, shape, allow_transpose=True)
            return {'op': 'read', 'slot': info['slot'], 'axes': axes}
        if node['kind'] == 'FloatingLiteral':
            number = float(node['value'])
            require(math.isfinite(number), 'guarded literal must be finite')
            return {'op': 'literal', 'value': number}
        if node['kind'] == 'UnaryOperator' and node.get('opcode') in {'+', '-'}:
            operand = expression(children(node)[0])
            return operand if node['opcode'] == '+' else {'op': 'neg', 'args': [operand]}
        require(node['kind'] == 'BinaryOperator' and node.get('opcode') in registry.binary
                and len(children(node)) == 2, 'unsupported guarded f32 expression')
        rule = registry.binary[node['opcode']]
        require(rule['spelling'] == node['opcode'], 'guarded operator spelling mismatch')
        return {'op': rule['intent'], 'spelling': rule['spelling'],
                'args': [expression(term) for term in children(node)]}

    return {'parameters': abi, 'shape': shape, 'store': target['slot'],
            'expression': expression(first_value)}


def register(registry):
    registry.stage('cuda.tail_pointwise', 'cuda_ast', 'pointwise', capture)
