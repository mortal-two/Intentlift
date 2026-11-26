"""Clang-based CUDA capture and checked flat member computation."""
import hashlib
import json
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
    while node["kind"] in {"ImplicitCastExpr", "ParenExpr", "CStyleCastExpr"}:
        require(len(children(node)) == 1, "unsupported cast")
        if node["kind"] == "CStyleCastExpr":
            require(node["type"]["qualType"] in {"int", "unsigned int"}, "unsupported explicit conversion")
        node = children(node)[0]
    return node


def capture(parsed, context, registry):
    fn, contract = parsed["function"], parsed["contract"]
    parameters = [p for p in children(fn) if p["kind"] == "ParmVarDecl"]
    abi = contract["parameters"]
    require(len(parameters) == len(abi), "CUDA ABI does not match declared parameters", "input_invalid")
    bindings = {}
    for i, (p, info) in enumerate(zip(parameters, abi)):
        ty = p["type"]["qualType"].replace("__restrict", "").strip()
        require(ty in {"const float *", "float *"}, "initial capture requires f32 pointer parameters")
        require(info["dtype"] == "f32" and len(info["shape"]) == 1, "initial capture requires rank-one f32")
        require(info["name"] == p["name"], "CUDA parameter order/name mismatch", "input_invalid")
        require(info["mode"] in {"input", "output", "inout"}, "unsupported ABI effect")
        require(not (ty == "const float *" and info["mode"] != "input"), "const parameter cannot be output", "input_invalid")
        bindings[p["name"]] = {**info, "slot": i}
    shapes = {tuple(p["shape"]) for p in abi}
    require(len(shapes) == 1, "flat pointwise views must have equal extents")
    n = next(iter(shapes))[0]
    require(0 < n <= 1024 and contract["grid"] == [1, 1, 1]
            and contract["block"] == [n, 1, 1] and contract["shared_mem"] == 0,
            "threadIdx.x member map requires one block covering the complete extent")
    body = next(p for p in children(fn) if p["kind"] == "CompoundStmt")
    statements = children(body)
    require(len(statements) == 1 and statements[0]["kind"] == "BinaryOperator"
            and statements[0]["opcode"] == "=", "initial capture requires one total output assignment")

    def access(node, store=False):
        node = unwrap(node)
        require(node["kind"] == "ArraySubscriptExpr", "expected direct member access")
        base, index = children(node)
        base = unwrap(base)
        require(base["kind"] == "DeclRefExpr", "indirect or offset pointer access requires extension")
        name = base["referencedDecl"]["name"]
        require(name in bindings, "access must refer to a parameter")
        binding = bindings[name]
        require(binding["mode"] in ({"output", "inout"} if store else {"input", "inout"}),
                "access effect differs from trusted interface", "input_invalid")
        index = unwrap(index)
        require(index["kind"] == "PseudoObjectExpr", "unsupported CUDA member index")
        refs = {p.get("referencedDecl", {}).get("name") for p in walk(index) if p["kind"] == "DeclRefExpr"}
        members = {p.get("name") for p in walk(index) if p["kind"] == "MemberExpr"}
        require(refs == {"threadIdx"} and members == {"__fetch_builtin_x"}, "expected exact threadIdx.x access")
        return binding["slot"]

    def expression(node):
        node = unwrap(node)
        require(node.get("type", {}).get("qualType") in {"float", "const float"}, "only f32 arithmetic is initially supported")
        if node["kind"] == "ArraySubscriptExpr":
            return {"op": "read", "slot": access(node)}
        require(node["kind"] == "BinaryOperator", "unhandled CUDA expression", node_kind=node["kind"])
        opcode = node["opcode"]
        require(opcode in registry.binary, "missing binary arithmetic capability", cuda_opcode=opcode,
                registration="Registry.binary_operator(cuda, intent, spelling)")
        rule = registry.binary[opcode]
        left, right = children(node)
        return {"op": rule["intent"], "spelling": rule["spelling"],
                "args": [expression(left), expression(right)]}

    left, right = children(statements[0])
    destination = access(left, store=True)
    outputs = [i for i, p in enumerate(abi) if p["mode"] != "input"]
    require(outputs == [destination], "all declared mutable views must be observed by the flat assignment")
    return {"parameters": abi, "extent": n, "store": destination, "expression": expression(right)}


def verify_ast(value):
    require(isinstance(value, dict) and "function" in value and "contract" in value, "invalid cuda_ast", "extension_invalid")


def verify_pointwise(value):
    require(isinstance(value, dict) and value.get("extent", 0) > 0 and "expression" in value,
            "invalid pointwise representation", "extension_invalid")


def register(registry):
    registry.representation("cuda_ast", verify_ast)
    registry.representation("pointwise", verify_pointwise)
    registry.stage("cuda.parse", "cuda_request", "cuda_ast", parse)
    registry.stage("cuda.pointwise", "cuda_ast", "pointwise", capture)
