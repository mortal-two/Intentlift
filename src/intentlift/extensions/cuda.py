"""Clang-based CUDA capture and checked pointwise member computation."""
import hashlib
import json
import math
import operator
from pathlib import Path
import subprocess
from intentlift.core.errors import require, LiftError


def parse(request, context, registry):
    path = Path(request["source"])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == request["source_sha256"],
            "source changed since Bench described it", "input_invalid")
    command = [request["clang"], "-x", "cuda", "--cuda-path=" + request["cuda_path"],
               "--cuda-gpu-arch=" + request["arch"], "--cuda-device-only", "-nocudalib",
               "-Wno-unknown-cuda-version", "-Wno-deprecated-literal-operator", "-std=c++17"]
    for directory in request["include_dirs"]:
        command += ["-I", directory]
    command += ["-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang",
                "-ast-dump-filter=" + request["entry"], str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    require(result.returncode == 0, "Clang could not parse CUDA", "source_compile_failed",
            diagnostic=result.stderr[-3000:])
    decoder = json.JSONDecoder()
    remaining = result.stdout.lstrip()
    definitions = []
    while remaining:
        node, end = decoder.raw_decode(remaining)
        if node.get("kind") == "FunctionDecl" and node.get("name") == request["entry"]:
            if any(child["kind"] == "CompoundStmt" for child in node.get("inner", [])):
                definitions.append(node)
        remaining = remaining[end:].lstrip()
    require(len(definitions) == 1, "expected exactly one selected CUDA definition", "input_invalid")
    return {"function": definitions[0], "contract": request}


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
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "&": operator.and_, "|": operator.or_, "^": operator.xor,
    "<<": operator.lshift, ">>": operator.rshift,
}
COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}


def qual_type(node):
    return node.get("type", {}).get("qualType")


def index_values(node, lanes, local=None):
    """Evaluate only defined 32-bit integer operations over the concrete launch."""
    kind, dtype = node["kind"], qual_type(node)
    require(dtype in {"int", "unsigned int"}, "unsupported CUDA index type", dtype=dtype)
    if kind == "PseudoObjectExpr":
        nodes = list(walk(node))
        refs = {(p.get("referencedDecl", {}).get("name"), qual_type(p))
                for p in nodes if p["kind"] == "DeclRefExpr"}
        members = {p.get("name") for p in nodes if p["kind"] == "MemberExpr"}
        require(refs == {("threadIdx", "const __cuda_builtin_threadIdx_t")}
                and members == {"__fetch_builtin_x"}, "expected CUDA threadIdx.x")
        return tuple(range(lanes))
    if kind == "IntegerLiteral":
        return (int(node["value"]),) * lanes
    if kind == "DeclRefExpr":
        require(local is not None and node["referencedDecl"]["id"] == local[0],
                "unsupported CUDA index variable")
        return (local[1],) * lanes
    if kind in {"ImplicitCastExpr", "CStyleCastExpr", "ParenExpr"}:
        require(kind == "ParenExpr" or node["castKind"] in {"NoOp", "IntegralCast", "LValueToRValue"},
                "unsupported CUDA index conversion")
        values = index_values(children(node)[0], lanes, local)
        if dtype == "unsigned int":
            return tuple(value % (1 << 32) for value in values)
        require(all(-(1 << 31) <= value < (1 << 31) for value in values),
                "index cast outside the signed 32-bit range")
        return values
    opcode = node.get("opcode")
    require(kind == "BinaryOperator" and opcode in INDEX_OPERATORS,
            "unsupported CUDA index expression", node_kind=kind, cuda_opcode=opcode)
    left, right = children(node)
    lhs, rhs = index_values(left, lanes, local), index_values(right, lanes, local)
    if opcode in {"<<", ">>"}:
        require(all(0 <= value < 32 for value in rhs) and all(value >= 0 for value in lhs),
                "unsupported CUDA shift operands")
    values = tuple(INDEX_OPERATORS[opcode](a, b) for a, b in zip(lhs, rhs))
    if dtype == "unsigned int":
        return tuple(value % (1 << 32) for value in values)
    require(all(-(1 << 31) <= value < (1 << 31) for value in values),
            "index operation exceeds the signed 32-bit range")
    return values


def member_axes(indices, shape, output_indices, output_shape, allow_transpose):
    identity = tuple(range(len(output_shape)))
    if shape == output_shape and indices == output_indices:
        return identity
    if allow_transpose and len(output_shape) == 2 and shape == output_shape[::-1]:
        rows, cols = output_shape
        if all(actual == (out % cols) * rows + out // cols
               for actual, out in zip(indices, output_indices)):
            return (1, 0)
    require(False, "CUDA access is not a checked member identity or transpose")


def assignment(node):
    require(node["kind"] == "BinaryOperator" and node.get("opcode") == "=",
            "expected a total output assignment")
    return children(node)


def literal_integer(node):
    node = unwrap(node)
    require(node["kind"] == "IntegerLiteral" and qual_type(node) == "int",
            "ordered CUDA loop requires an int literal bound")
    return int(node["value"])


def ordered_loop(loop):
    """Check zero-based int induction, exclusive literal bound and unit increment."""
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
    require(body["kind"] == "CompoundStmt" and len(children(body)) == 1,
            "ordered CUDA body must update one accumulator")
    return index["id"], limit, children(body)[0]


def half_to_float_operand(node):
    """Recognize the CUTLASS binary16 conversion without discarding any casts."""
    while node["kind"] == "ParenExpr" or (
        node["kind"] == "CStyleCastExpr" and node.get("castKind") == "NoOp" and qual_type(node) == "float"
    ):
        node = children(node)[0]
    require(node["kind"] == "ImplicitCastExpr" and node.get("castKind") == "UserDefinedConversion"
            and qual_type(node) == "float", "matrix factors must widen binary16 to f32")
    call = children(node)[0]
    require(call["kind"] == "CXXMemberCallExpr" and qual_type(call) == "float" and len(children(call)) == 1,
            "expected a CUTLASS half conversion")
    member = children(call)[0]
    require(member["kind"] == "MemberExpr" and member.get("name") == "operator float"
            and len(children(member)) == 1, "expected CUTLASS half operator float")
    operand = unwrap(children(member)[0])
    require(operand["kind"] == "ArraySubscriptExpr"
            and operand["type"].get("desugaredQualType", qual_type(operand))
            in {"const cutlass::half_t", "cutlass::half_t"},
            "binary16 conversion must read a CUTLASS half view")
    return operand


def float_to_int_operand(node):
    """Recognize a signed 32-bit float-to-int conversion of one loaded value."""
    explicit_casts = {"CStyleCastExpr", "CXXStaticCastExpr", "CXXFunctionalCastExpr"}
    while node["kind"] == "ParenExpr" or (
        node["kind"] in explicit_casts and node.get("castKind") == "NoOp" and qual_type(node) == "int"
    ):
        node = children(node)[0]
    require(node["kind"] in explicit_casts | {"ImplicitCastExpr"}
            and node.get("castKind") == "FloatingToIntegral" and qual_type(node) == "int",
            "expected a conversion from f32 to signed i32")
    operand = unwrap(children(node)[0])
    require(operand["kind"] == "ArraySubscriptExpr" and qual_type(operand) in {"float", "const float"},
            "i32 conversion requires a direct f32 input read")
    return operand


def capture_matrix_product(statements, shape, target_indices, access):
    """Verify the ordered contraction against all output lanes and reduction positions."""
    declaration, initialization, loop, store = statements
    require(len(shape) == 2 and declaration["kind"] == "DeclStmt" and len(children(declaration)) == 1,
            "ordered matrix product requires one thread-local f32 accumulator")
    acc = children(declaration)[0]
    require(acc["kind"] == "VarDecl" and qual_type(acc) == "float[1]"
            and not children(acc) and not acc.get("storageClass"),
            "expected one automatic f32 accumulator cell")

    def is_accumulator(node):
        node = unwrap(node)
        if node["kind"] != "ArraySubscriptExpr":
            return False
        base, index = children(node)
        base, index = unwrap(base), unwrap(index)
        return (base["kind"] == "DeclRefExpr" and base["referencedDecl"]["id"] == acc["id"]
                and index["kind"] == "IntegerLiteral" and int(index["value"]) == 0)

    dest, initial = assignment(initialization)
    initial = unwrap(initial)
    require(is_accumulator(dest) and initial["kind"] == "FloatingLiteral" and qual_type(initial) == "float",
            "ordered matrix product requires a constant f32 initializer")
    initial_value = float(initial["value"])
    require(math.isfinite(initial_value), "unsupported ordered accumulator initializer")
    _, result = assignment(store)
    require(is_accumulator(result), "ordered matrix output must store the final accumulator")
    iv, limit, update = ordered_loop(loop)
    dest, value = assignment(update)
    value = unwrap(value)
    require(is_accumulator(dest) and value["kind"] == "BinaryOperator"
            and value.get("opcode") == "+" and qual_type(value) == "float"
            and is_accumulator(children(value)[0]), "expected accumulator plus a separately rounded f32 product")
    product = unwrap(children(value)[1])
    require(product["kind"] == "BinaryOperator" and product.get("opcode") == "*" and qual_type(product) == "float",
            "expected f32 multiplication in ordered matrix loop")

    rows, cols = shape
    factors = []
    for term in children(product):
        operand = half_to_float_operand(term)
        binding, _ = access(operand, local=(iv, 0))
        require(binding["dtype"] == "f16" and binding["mode"] == "input", "expected a read-only f16 matrix")
        factor_shape = tuple(binding["shape"])
        row_factor = factor_shape == (rows, limit)
        col_factor = factor_shape == (limit, cols)
        for k in range(limit):
            _, observed = access(operand, local=(iv, k))
            row_factor = row_factor and all(index == (out // cols) * limit + k
                                            for index, out in zip(observed, target_indices))
            col_factor = col_factor and all(index == k * cols + out % cols
                                            for index, out in zip(observed, target_indices))
            require(row_factor or col_factor, "ordered matrix read does not match an in-bounds contraction")
        factors.append((binding["slot"], row_factor, col_factor))
    first, second = factors
    if first[1] and second[2]:
        axes = ["row", "col"]
    else:
        require(first[2] and second[1], "ordered matrix product requires one row and one column factor")
        axes = ["col", "row"]
    return {"op": "ordered_product", "initial": initial_value,
            "factors": [{"slot": factor[0], "axis": axis} for factor, axis in zip(factors, axes)]}


def split_body(statements):
    """Recognize a single store, optionally fed by a total local selection."""
    require(len(statements) in {1, 3}, "expected a total output assignment")
    destination, value = assignment(statements[-1])
    if len(statements) == 1:
        return destination, value, None
    declaration, branch = statements[:2]
    require(declaration["kind"] == "DeclStmt" and len(children(declaration)) == 1
            and branch["kind"] == "IfStmt" and len(children(branch)) == 3,
            "expected a local followed by an if/else with both values defined")
    local = children(declaration)[0]
    require(local["kind"] == "VarDecl" and qual_type(local) == "float"
            and not children(local) and not local.get("storageClass"),
            "selection local must be an uninitialized automatic f32")

    def is_local(node):
        node = unwrap(node)
        return node["kind"] == "DeclRefExpr" and node["referencedDecl"]["id"] == local["id"]

    def assigned_value(node):
        if node["kind"] == "CompoundStmt":
            require(len(children(node)) == 1, "selection branch must assign once")
            node = children(node)[0]
        require(node["kind"] == "BinaryOperator" and node.get("opcode") == "="
                and is_local(children(node)[0]), "selection branch must assign the same local")
        return children(node)[1]

    require(is_local(value), "output assignment must use the selected local")
    cond, yes, no = children(branch)
    return destination, None, (cond, assigned_value(yes), assigned_value(no))


def capture(parsed, context, registry):
    fn, contract = parsed["function"], parsed["contract"]
    parameters = [p for p in children(fn) if p["kind"] == "ParmVarDecl"]
    abi = contract["parameters"]
    require(len(parameters) == len(abi), "CUDA ABI does not match declared parameters", "input_invalid")
    bindings = {}
    for i, (p, info) in enumerate(zip(parameters, abi)):
        ty = qual_type(p).replace("__restrict", "").strip()
        pointer_types = {"float *": "f32", "half_t *": "f16", "cutlass::half_t *": "f16", "int *": "i32"}
        require(pointer_types.get(ty.removeprefix("const ")) == info["dtype"],
                "CUDA pointer type differs from supported storage type", dtype=info["dtype"])
        require(len(info["shape"]) in {1, 2}
                and all(type(dim) is int and dim > 0 for dim in info["shape"]),
                "capture requires positive static rank-one or rank-two views")
        require(info["name"] == p["name"], "CUDA parameter order/name mismatch", "input_invalid")
        require(info["mode"] in {"input", "output", "inout"}, "unsupported ABI effect")
        require(not (ty.startswith("const ") and info["mode"] != "input"), "const parameter cannot be output", "input_invalid")
        require(info["dtype"] != "f16" or info["mode"] == "input", "binary16 support requires read-only views")
        bindings[p["id"]] = {**info, "slot": i}
    outputs = [i for i, p in enumerate(abi) if p["mode"] != "input"]
    require(len(outputs) == 1, "pointwise capture requires one observable view")
    shape = tuple(abi[outputs[0]]["shape"])
    n = math.prod(shape)
    require(0 < n <= 1024 and contract["grid"] == [1, 1, 1]
            and contract["block"] == [n, 1, 1] and contract["shared_mem"] == 0,
            "threadIdx.x member map requires one block covering the complete extent")
    body = next(p for p in children(fn) if p["kind"] == "CompoundStmt")
    statements = children(body)
    ordered_product = len(statements) == 4 and statements[2]["kind"] == "ForStmt"
    if ordered_product:
        destination, _ = assignment(statements[-1])
    else:
        destination, value, selection = split_body(statements)

    def access(node, store=False, local=None):
        node = unwrap(node)
        require(node["kind"] == "ArraySubscriptExpr", "expected direct member access")
        base, index = children(node)
        base = unwrap(base)
        require(base["kind"] == "DeclRefExpr", "indirect or offset pointer access requires extension")
        declaration = base["referencedDecl"]["id"]
        require(declaration in bindings, "access must refer to a parameter")
        binding = bindings[declaration]
        require(binding["mode"] in ({"output", "inout"} if store else {"input", "inout"}),
                "access effect differs from trusted interface", "input_invalid")
        return binding, index_values(index, n, local)

    target, target_indices = access(destination, store=True)
    require(set(target_indices) == set(range(n)), "CUDA store must cover every output member exactly once")
    program = {"parameters": abi, "shape": shape, "store": target["slot"]}

    def read(node):
        binding, indices = access(node)
        require(binding["dtype"] == "f32", "pointwise value must come from an f32 view")
        axes = member_axes(indices, tuple(binding["shape"]), target_indices, shape,
                           allow_transpose=binding["mode"] == "input")
        return {"op": "read", "slot": binding["slot"], "axes": axes}

    if target["dtype"] == "i32":
        require(not ordered_product and selection is None,
                "i32 output requires one direct f32 conversion per member")
        operand = read(float_to_int_operand(value))
        return {**program, "expression": {"op": "cast", "dtype": "i32", "args": [operand]}}
    if ordered_product:
        return {**program, "expression": capture_matrix_product(statements, shape, target_indices, access)}

    def predicate(node):
        node = unwrap(node)
        require(node["kind"] == "BinaryOperator" and node.get("opcode") in COMPARISONS
                and qual_type(node) == "bool", "expected an f32 comparison for member selection")
        return {"op": "compare", "spelling": node["opcode"],
                "args": [expression(child) for child in children(node)]}

    def select(cond, yes, no):
        return {"op": "select", "args": [predicate(cond), expression(yes), expression(no)]}

    def expression(node):
        node = unwrap(node)
        require(qual_type(node) in {"float", "const float"}, "pointwise expression must be f32")
        kind = node["kind"]
        if kind == "ArraySubscriptExpr":
            return read(node)
        if kind == "FloatingLiteral":
            value = float(node["value"])
            require(math.isfinite(value), "unsupported non-finite CUDA f32 literal")
            return {"op": "literal", "value": value}
        if kind == "UnaryOperator" and node["opcode"] in {"+", "-"}:
            operand = expression(children(node)[0])
            if node["opcode"] == "-" and operand["op"] == "literal":
                return {"op": "literal", "value": -operand["value"]}
            return operand if node["opcode"] == "+" else {"op": "neg", "args": [operand]}
        if kind == "ConditionalOperator":
            return select(*children(node))
        require(kind == "BinaryOperator", "unhandled CUDA expression", node_kind=kind)
        opcode = node["opcode"]
        require(opcode in registry.binary, "missing binary arithmetic capability", cuda_opcode=opcode,
                registration="Registry.binary_operator(cuda, intent, spelling)")
        rule = registry.binary[opcode]
        left, right = children(node)
        return {"op": rule["intent"], "spelling": rule["spelling"],
                "args": [expression(left), expression(right)]}

    return {**program, "expression": select(*selection) if selection is not None else expression(value)}


def verify_ast(value):
    require(isinstance(value, dict) and "function" in value and "contract" in value, "invalid cuda_ast", "extension_invalid")


def verify_pointwise(value):
    require(isinstance(value, dict) and value.get("shape") and "expression" in value,
            "invalid pointwise representation", "extension_invalid")


def register(registry):
    registry.representation("cuda_ast", verify_ast)
    registry.representation("pointwise", verify_pointwise)
    registry.stage("cuda.parse", "cuda_request", "cuda_ast", parse)
    registry.stage("cuda.pointwise", "cuda_ast", "pointwise", capture)
