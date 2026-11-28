"""Checked eight-bit member copies and byte-to-i32 arithmetic."""
import math
from intentlift.core.errors import require
from . import cuda, f16_pointwise
from ._cuda_ast import assignment, children, index_values, qual_type, unwrap
_BYTE_TYPES = {'i8': 'signed char *', 'u8': 'unsigned char *'}
_BYTE_VALUES = {'i8': 'signed char', 'u8': 'unsigned char'}


def canonical_type(node):
    return node['type'].get('desugaredQualType', qual_type(node))


def capture(parsed, context, registry):
    abi = parsed['contract']['parameters']
    outputs = [i for i, p in enumerate(abi) if p['mode'] != 'input']
    mixed = bool(len(outputs) == 1 and abi[outputs[0]]['dtype'] == 'i32'
                 and any(p['dtype'] in _BYTE_TYPES for p in abi))
    if not mixed and (not abi or len({p['dtype'] for p in abi}) != 1 or abi[0]['dtype'] not in _BYTE_TYPES):
        return f16_pointwise.capture(parsed, context, registry)
    if mixed:
        require(len(abi) >= 3 and all(p['dtype'] in _BYTE_TYPES and p['mode'] == 'input'
                    if i != outputs[0] else p['dtype'] == 'i32' and p['mode'] == 'output'
                    for i, p in enumerate(abi)), 'widened arithmetic requires byte inputs and i32 output')
    else:
        require(len(abi) == 2 and len(outputs) == 1, 'byte member copy needs one source and one destination')
    fn, contract = parsed['function'], parsed['contract']
    decls = [node for node in children(fn) if node['kind'] == 'ParmVarDecl']
    require(len(decls) == len(abi), 'CUDA ABI parameter count mismatch', 'input_invalid')
    bindings = {}
    for slot, (decl, info) in enumerate(zip(decls, abi)):
        ty = canonical_type(decl).replace('__restrict', '').strip()
        expected = ('int *' if slot == outputs[0] else _BYTE_TYPES[info['dtype']]) if mixed else _BYTE_TYPES[abi[0]['dtype']]
        require(ty.removeprefix('const ') == expected and info['name'] == decl['name']
                and info['mode'] in {'input', 'output'}
                and (info['mode'] != 'output' or not ty.startswith('const '))
                and len(info['shape']) in {1, 2}
                and all(type(d) is int and d > 0 for d in info['shape']),
                'byte member requires compatible views', ty=ty)
        bindings[decl['id']] = {**info, 'slot': slot}
    shape = tuple(abi[outputs[0]]['shape'])
    n = math.prod(shape)
    require(0 < n <= 1024 and contract['grid'] == [1, 1, 1]
            and contract['block'] == [n, 1, 1] and contract['shared_mem'] == 0,
            'byte member requires a complete single block')
    body = next(node for node in children(fn) if node['kind'] == 'CompoundStmt')
    require(len(children(body)) == 1, 'byte member requires one total assignment')
    target, value = assignment(children(body)[0])

    def access(node, mode):
        node = unwrap(node)
        require(node['kind'] == 'ArraySubscriptExpr', 'byte member must be directly indexed')
        base, offset = children(node)
        base = unwrap(base)
        require(base['kind'] == 'DeclRefExpr' and base['referencedDecl']['id'] in bindings,
                'byte member must refer to an ABI parameter')
        info = bindings[base['referencedDecl']['id']]
        require(info['mode'] == mode, 'byte member access effect mismatch')
        return info, index_values(offset, n)

    dest, targets = access(target, 'output')
    require(dest['slot'] == outputs[0] and sorted(targets) == list(range(n)),
            'byte output must cover every member exactly once')

    def read(node):
        src, reads = access(node, 'input')
        axes = cuda.member_axes(reads, tuple(src['shape']), targets, shape, allow_transpose=True)
        return {'op': 'read', 'slot': src['slot'], 'axes': axes}, src

    if not mixed:
        expression, _ = read(value)
    else:
        value = unwrap(value)
        opcode = value.get('opcode')
        require(value['kind'] == 'BinaryOperator' and opcode in {'+', '*'}
                and qual_type(value) == 'int' and len(children(value)) == 2
                and opcode in registry.binary and registry.binary[opcode]['spelling'] == opcode,
                'widened arithmetic requires registered int addition or multiplication')

        def promoted(node):
            widened = False
            top_type = qual_type(node)
            path = []
            while node['kind'] in {'ParenExpr', 'CStyleCastExpr', 'ImplicitCastExpr'}:
                if node['kind'] == 'ParenExpr':
                    require(len(children(node)) == 1, 'unexpected byte parentheses')
                else:
                    kind = node.get('castKind')
                    require(kind in {'IntegralCast', 'NoOp', 'LValueToRValue'}
                            and len(children(node)) == 1, 'unexpected byte conversion', cast=kind)
                    path.append((kind, qual_type(node)))
                    if kind == 'IntegralCast':
                        require(qual_type(node) == 'int' and canonical_type(children(node)[0]).removeprefix('const ') in _BYTE_VALUES.values(),
                                'byte promotion must be byte to int')
                        widened = True
                node = children(node)[0]
            require(top_type == 'int' and widened and node['kind'] == 'ArraySubscriptExpr',
                    'byte operand must be widened to int', top=top_type,
                    terminal=qual_type(node), terminal_kind=node['kind'], path=path)
            member, src = read(node)
            require(canonical_type(node).removeprefix('const ') == _BYTE_VALUES[src['dtype']],
                    'byte promotion type must agree with ABI', actual=canonical_type(node))
            return {'op': 'cast', 'dtype': 'i32', 'args': [member]}

        expression = {'op': registry.binary[opcode]['intent'], 'spelling': opcode,
                      'args': [promoted(term) for term in children(value)]}
    return {'parameters': abi, 'shape': shape, 'store': dest['slot'], 'expression': expression}


def register(registry):
    registry.stage('cuda.signed_byte_pointwise', 'cuda_ast', 'pointwise', capture)
