"""Checked Clang AST operations for CUDA capture stages."""
import operator

from intentlift.core.errors import require


def children(node):
    return node.get("inner", [])

def walk(node):
    yield node
    for child in children(node):
        yield from walk(child)

def unwrap(node):
    while node["kind"] in {"ImplicitCastExpr", "ParenExpr"}:
        require(len(children(node)) == 1, "unsupported cast")
        if node["kind"] == "ImplicitCastExpr":
            require(node["castKind"] in {"LValueToRValue", "ArrayToPointerDecay", "NoOp"},
                    "unsupported implicit conversion", cast_kind=node["castKind"])
        node = children(node)[0]
    return node

INDEX_OPERATORS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "%": operator.mod,
    "&": operator.and_, "|": operator.or_, "^": operator.xor,
    "<<": operator.lshift, ">>": operator.rshift,
}

COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}

INDEX_COMPARE = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
                 "==": operator.eq, "!=": operator.ne}

def qual_type(node):
    return node.get("type", {}).get("qualType")

def index_values(node, lanes, local=None, block=None):
    """Evaluate only defined 32-bit integer operations over the concrete launch."""
    kind, dtype = node["kind"], qual_type(node)
    require(dtype in {"int", "unsigned int"}, "unsupported CUDA index type", dtype=dtype)
    if kind == "PseudoObjectExpr":
        nodes = list(walk(node))
        refs = {(p.get("referencedDecl", {}).get("name"), qual_type(p))
                for p in nodes if p["kind"] == "DeclRefExpr"}
        members = {p.get("name") for p in nodes if p["kind"] == "MemberExpr"}
        if refs == {("threadIdx", "const __cuda_builtin_threadIdx_t")} and members == {"__fetch_builtin_x"}:
            return tuple(range(lanes))
        require(block is not None and refs == {("blockIdx", "const __cuda_builtin_blockIdx_t")}
                and len(members) == 1, "expected a checked CUDA launch index")
        axis = next(iter(members))
        require(axis in {"__fetch_builtin_x", "__fetch_builtin_y", "__fetch_builtin_z"},
                "unknown CUDA block axis")
        return (block["xyz".index(axis[-1])],) * lanes
    if kind == "IntegerLiteral":
        return (int(node["value"]),) * lanes
    if kind == "DeclRefExpr":
        require(local is not None and node["referencedDecl"]["id"] == local[0],
                "unsupported CUDA index variable")
        return (local[1],) * lanes
    if kind in {"ImplicitCastExpr", "CStyleCastExpr", "ParenExpr"}:
        require(kind == "ParenExpr" or node["castKind"] in {"NoOp", "IntegralCast", "LValueToRValue"},
                "unsupported CUDA index conversion")
        values = index_values(children(node)[0], lanes, local, block)
        if dtype == "unsigned int":
            return tuple(value % (1 << 32) for value in values)
        require(all(-(1 << 31) <= value < (1 << 31) for value in values),
                "index cast outside the signed 32-bit range")
        return values
    opcode = node.get("opcode")
    require(kind == "BinaryOperator" and opcode in INDEX_OPERATORS,
            "unsupported CUDA index expression", node_kind=kind, cuda_opcode=opcode)
    left, right = children(node)
    lhs, rhs = index_values(left, lanes, local, block), index_values(right, lanes, local, block)
    if opcode in {"<<", ">>"}:
        require(all(0 <= value < 32 for value in rhs) and all(value >= 0 for value in lhs),
                "unsupported CUDA shift operands")
    if opcode == "%":
        require(all(value > 0 for value in rhs) and all(value >= 0 for value in lhs),
                "unsupported CUDA remainder operands")
    values = tuple(INDEX_OPERATORS[opcode](a, b) for a, b in zip(lhs, rhs))
    if dtype == "unsigned int":
        return tuple(value % (1 << 32) for value in values)
    require(all(-(1 << 31) <= value < (1 << 31) for value in values),
            "index operation exceeds the signed 32-bit range")
    return values

def assignment(node):
    require(node["kind"] == "BinaryOperator" and node.get("opcode") == "=",
            "expected a total output assignment")
    return children(node)

def literal_integer(node):
    node = unwrap(node)
    require(node["kind"] == "IntegerLiteral" and qual_type(node) == "int",
            "ordered CUDA loop requires an int literal bound")
    return int(node["value"])

def ordered_body(loop):
    """Check zero-based int induction, exclusive literal bound and unit increment."""
    if loop["kind"] == "AttributedStmt":
        require(len(children(loop)) == 2 and children(loop)[0]["kind"] == "LoopHintAttr",
                "unsupported CUDA loop attributes")
        loop = children(loop)[1]
    require(loop["kind"] == "ForStmt", "expected an ordered CUDA loop")
    parts = children(loop)
    require(len(parts) == 5 and parts[1] == {}, "unsupported CUDA for-loop structure")
    init, _, condition, step, body = parts
    require(init["kind"] == "DeclStmt" and len(children(init)) == 1,
            "expected one ordered loop index")
    index = children(init)[0]
    require(index["kind"] == "VarDecl" and qual_type(index) == "int"
            and not index.get("storageClass") and len(children(index)) == 1
            and literal_integer(children(index)[0]) == 0, "expected a zero-based int loop index")

    def is_index(node):
        node = unwrap(node)
        return node["kind"] == "DeclRefExpr" and node["referencedDecl"]["id"] == index["id"]

    require(condition["kind"] == "BinaryOperator" and condition.get("opcode") == "<"
            and is_index(children(condition)[0])
            and step["kind"] == "UnaryOperator" and step.get("opcode") == "++"
            and is_index(children(step)[0]), "expected an increasing exclusive loop")
    limit = literal_integer(children(condition)[1])
    require(0 < limit <= 1024, "ordered CUDA loop bound is outside the checked static range")
    require(body["kind"] == "CompoundStmt", "expected a compound CUDA loop body")
    return index["id"], limit, children(body)

def ordered_loop(loop):
    index, limit, body = ordered_body(loop)
    require(len(body) == 1, "ordered CUDA body must update one accumulator")
    return index, limit, body[0]

def declaration(stmt, valid_types):
    require(stmt["kind"] == "DeclStmt" and len(children(stmt)) == 1, "expected one CUDA declaration")
    decl = children(stmt)[0]
    require(decl["kind"] == "VarDecl" and qual_type(decl) in valid_types,
            "unexpected CUDA local type", dtype=qual_type(decl))
    return decl

def ref_is(node, decl_id):
    return node["kind"] == "DeclRefExpr" and node.get("referencedDecl", {}).get("id") == decl_id

def subscript(node):
    node = unwrap(node)
    require(node["kind"] == "ArraySubscriptExpr", "expected an array access")
    base, index = children(node)
    return unwrap(base), index

def callee(node):
    require(node["kind"] == "CallExpr" and children(node), "expected a CUDA intrinsic call")
    function, *args = children(node)
    require(function["kind"] == "ImplicitCastExpr" and len(children(function)) == 1,
            "expected a direct CUDA function call")
    reference = children(function)[0]
    require(reference["kind"] == "DeclRefExpr", "expected a direct CUDA callee")
    return reference, args

def source_spelling(node, original):
    begin, end = (node["range"][key] for key in ("begin", "end"))
    require("offset" in begin and "offset" in end and "tokLen" in end,
            "CUDA template must have an explicit source spelling")
    return b"".join(original[begin["offset"]:end["offset"] + end["tokLen"]].split())

def guard(node, lanes, block=None):
    if node["kind"] != "IfStmt":
        return tuple(range(lanes)), [node]
    require(len(children(node)) == 2, "output guard must have no else or side effect")
    condition, body = children(node)
    condition = unwrap(condition)
    op = condition.get("opcode")
    require(condition["kind"] == "BinaryOperator" and op in INDEX_COMPARE
            and body["kind"] == "CompoundStmt", "unsupported CUDA output guard")
    lhs, rhs = (index_values(arg, lanes, block=block) for arg in children(condition))
    active = tuple(i for i, (a, b) in enumerate(zip(lhs, rhs)) if INDEX_COMPARE[op](a, b))
    return active, children(body)
