"""Checked signed-integer pointwise additions with exact-width operand storage."""
import math
from intentlift.core.errors import require
from . import cuda
from ._cuda_ast import assignment, children, index_values, qual_type, unwrap


SIGNED_POINTERS = {"i32": {"int *"}, "i64": {"int64_t *", "long *", "long long *"}}
SIGNED_VALUES = {"i32": {"int"}, "i64": {"int64_t", "long", "long long"}}


def capture(parsed, context, registry):
    if 'staged_bf16' in parsed:
        return parsed['staged_bf16']
    if any(key.startswith("staged_") for key in parsed):
        return cuda.capture(parsed, context, registry)
    fn, contract = parsed["function"], parsed["contract"]
    abi = contract["parameters"]
    outputs = [i for i, p in enumerate(abi) if p["mode"] != "input"]
    if (len(outputs) != 1 or abi[outputs[0]]["dtype"] not in SIGNED_POINTERS
            or not any(p["dtype"] == abi[outputs[0]]["dtype"] and p["mode"] == "input" for p in abi)):
        return cuda.capture(parsed, context, registry)

    dtype = abi[outputs[0]]["dtype"]
    parameters = [p for p in children(fn) if p["kind"] == "ParmVarDecl"]
    require(len(parameters) == len(abi), "CUDA ABI does not match declared parameters", "input_invalid")
    bindings = {}
    for slot, (decl, info) in enumerate(zip(parameters, abi)):
        ty = decl["type"].get("desugaredQualType", qual_type(decl)).replace("__restrict", "").strip()
        require(info["dtype"] == dtype and ty.removeprefix("const ") in SIGNED_POINTERS[dtype]
                and info["name"] == decl["name"] and info["mode"] in {"input", "output"}
                and (not ty.startswith("const ") or info["mode"] == "input")
                and len(info["shape"]) in {1, 2}
                and all(type(dim) is int and dim > 0 for dim in info["shape"]),
                "integer member capture requires compatible signed pointer views", ty=ty)
        bindings[decl['id']] = {**info, 'slot': slot}
    shape = tuple(abi[outputs[0]]["shape"])
    n = math.prod(shape)
    require(0 < n <= 1024 and contract["grid"] == [1, 1, 1]
            and contract["block"] == [n, 1, 1] and contract["shared_mem"] == 0,
            "integer member capture requires one full one-dimensional block")
    body = next(p for p in children(fn) if p["kind"] == "CompoundStmt")
    statements = children(body)
    require(len(statements) == 1, "integer member capture requires a single total store")
    dest, rhs = assignment(statements[0])

    def access(node, mode):
        node = unwrap(node)
        require(node["kind"] == "ArraySubscriptExpr", "integer access must be a direct subscript")
        base, index = children(node)
        base = unwrap(base)
        require(base["kind"] == "DeclRefExpr" and base["referencedDecl"]["id"] in bindings,
                "integer access must use a parameter")
        binding = bindings[base["referencedDecl"]["id"]]
        require(binding["mode"] == mode, "integer member access has an incompatible effect")
        return binding, index_values(index, n)

    target, target_indices = access(dest, "output")
    require(target["slot"] == outputs[0] and sorted(target_indices) == list(range(n)),
            "integer stores must cover every output member exactly once")
    rhs = unwrap(rhs)
    rhs_type = rhs["type"].get("desugaredQualType", qual_type(rhs))
    require(rhs["kind"] == "BinaryOperator" and rhs.get("opcode") == "+"
            and rhs_type in SIGNED_VALUES[dtype] and "+" in registry.binary,
            "integer member expression requires same-width signed addition", rhs_type=rhs_type)
    operands = []
    for term in children(rhs):
        binding, indices = access(term, "input")
        axes = cuda.member_axes(indices, tuple(binding["shape"]), target_indices, shape,
                                allow_transpose=True)
        operands.append({"op": "read", "slot": binding["slot"], "axes": axes})
    rule = registry.binary["+"]
    require(rule["spelling"] == "+", "signed addition requires an addition operator")
    return {"parameters": abi, "shape": shape, "store": target["slot"],
            "expression": {"op": rule["intent"], "spelling": "+", "args": operands}}


def register(registry):
    registry.stage("cuda.integer_pointwise", "cuda_ast", "pointwise", capture)
