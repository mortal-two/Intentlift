"""Lift an ordered float32 RMS reduction with checked packed readback."""
import math
from pathlib import Path

from intentlift.core.errors import require
from ._cuda_ast import (
    assignment, callee, children, declaration, guard, index_values,
    ordered_body, ordered_loop, qual_type, ref_is, source_spelling, unwrap, walk,
)
from .convolution import address_operand
from .cuda import parameter_access, private_at, scalar_literal, unwrapped_call, vector_pointer


PACK = "ulonglong4"
PACK_MEMBERS = tuple((word, lane) for word in "xyzw" for lane in "xy")


def operand(node, kind, dtype, opcode=None, cast=None):
    node = unwrap(node)
    parts = children(node)
    require(node["kind"] == kind and qual_type(node) == dtype and len(parts) == 1
            and (opcode is None or node.get("opcode") == opcode)
            and (cast is None or node.get("castKind") == cast),
            "unexpected packed RMS operand", node_kind=node["kind"], dtype=qual_type(node))
    return parts[0]


def packed_component(node, storage, outer, inner):
    node = unwrap(node)
    require(node["kind"] == "MemberExpr" and node.get("name") == inner
            and node.get("isArrow") and qual_type(node) == "float" and len(children(node)) == 1,
            "expected a float2 lane in the packed RMS result")
    cast = operand(children(node)[0], "CStyleCastExpr", "float2 *", cast="BitCast")
    member = unwrap(operand(cast, "UnaryOperator", "unsigned long long *", opcode="&"))
    require(member["kind"] == "MemberExpr" and member.get("name") == outer
            and not member.get("isArrow") and qual_type(member) == "unsigned long long"
            and len(children(member)) == 1 and ref_is(unwrap(children(member)[0]), storage),
            "packed RMS loop moves a value between vector lanes or uses a different vector")


def intrinsic(node, name, result_type, arity, original):
    node = unwrapped_call(node)
    function, args = callee(node)
    require(qual_type(node) == result_type and len(args) == arity
            and source_spelling(function, original) == name,
            "unexpected RMS arithmetic or vector intrinsic", expected=name.decode())
    return args


def scale_epsilon(node, acc, width, original):
    value, = intrinsic(node, b"rsqrtf", "float", 1, original)
    value = unwrap(value)
    require(value["kind"] == "BinaryOperator" and value.get("opcode") == "+" and qual_type(value) == "float",
            "RMS scale must add epsilon before rsqrtf")
    quotient, epsilon = children(value)
    quotient = unwrap(quotient)
    require(quotient["kind"] == "BinaryOperator" and quotient.get("opcode") == "/"
            and qual_type(quotient) == "float", "RMS mean square must divide before adding epsilon")
    total, divisor = children(quotient)
    require(private_at(total, acc) and scalar_literal(divisor, "float") == width,
            "RMS divisor does not match the reduced row width")
    epsilon = scalar_literal(epsilon, "float")
    require(epsilon > 0, "RMS epsilon must be positive")
    return epsilon


def capture(parsed, context, registry):
    function, contract = parsed["function"], parsed["contract"]
    body = next(node for node in children(function) if node["kind"] == "CompoundStmt")
    calls = {node.get("referencedDecl", {}).get("name") for node in walk(body)
             if node.get("kind") == "DeclRefExpr"}
    if not {"rsqrtf", "load_global_256", "store_global_256"} <= calls:
        return parsed

    params = contract["parameters"]
    arguments = [node for node in children(function) if node["kind"] == "ParmVarDecl"]
    require(len(params) == len(arguments) == 2
            and sorted(p["mode"] for p in params) == ["input", "output"]
            and all(p["dtype"] == "f32" and len(p["shape"]) == 2 and p["shape"] == params[0]["shape"]
                    for p in params), "RMS capture needs same-shape rank-two f32 input and output")
    require(all(p["name"] == node["name"] and qual_type(node).replace("__restrict", "").strip()
                == ("const float *" if p["mode"] == "input" else "float *")
                for p, node in zip(params, arguments)), "RMS CUDA ABI mismatch")
    slots = {node["id"]: p["position"] for p, node in zip(params, arguments)}
    source, output = (next(p["position"] for p in params if p["mode"] == mode)
                      for mode in ("input", "output"))
    rows, width = params[output]["shape"]
    lanes = contract["block"][0]
    require(rows > 0 and width > 0 and width % len(PACK_MEMBERS) == 0
            and contract["grid"] == [1, 1, 1] and contract["block"][1:] == [1, 1]
            and 0 < lanes <= 1024 and contract["shared_mem"] == 0,
            "unsupported launch for packed RMS capture")

    statements = children(body)
    require(len(statements) == 4 and statements[-1]["kind"] == "IfStmt",
            "expected RMS accumulation followed by guarded vector output")
    local, init, reduce, output_guard = statements
    acc = declaration(local, {"float[1]"})
    require(not children(acc) and not acc.get("storageClass"), "RMS accumulator must be thread private")
    acc_id = acc["id"]
    target, value = assignment(init)
    initial = scalar_literal(value, "float")
    require(private_at(target, acc_id) and initial == 0.0 and math.copysign(1.0, initial) > 0,
            "RMS accumulator must start at positive zero")
    reduction, count, update = ordered_loop(reduce)
    target, value = assignment(update)
    value = unwrap(value)
    require(count == width and private_at(target, acc_id)
            and value["kind"] == "BinaryOperator" and value.get("opcode") == "+"
            and qual_type(value) == "float", "RMS must accumulate one complete row in f32")
    previous, square = children(value)
    square = unwrap(square)
    require(private_at(previous, acc_id) and square["kind"] == "BinaryOperator"
            and square.get("opcode") == "*" and qual_type(square) == "float",
            "RMS must accumulate the ordered square of each input")

    original = Path(contract["source"]).read_bytes()
    active, guarded = guard(output_guard, lanes)
    require(len(guarded) == 1 and len(active) == rows, "expected only one output owner per RMS row")
    tile, tiles, block = ordered_body(guarded[0])
    require(tiles * len(PACK_MEMBERS) == width and len(block) == 4 + len(PACK_MEMBERS),
            "packed RMS writes must cover each row with complete vectors")
    dest, loaded, scale = (declaration(stmt, {PACK}) for stmt in block[:3])
    require(all(not node.get("storageClass") and len(children(node)) == 1 for node in (dest, loaded, scale)),
            "expected three private packed RMS vectors")
    default, = children(dest)
    require(default["kind"] == "CXXConstructExpr" and not children(default),
            "RMS output vector has an unexpected initializer")
    load_addr, = intrinsic(children(loaded)[0], b"tl::load_global_256", PACK, 1, original)
    store_addr, stored_value = intrinsic(block[-1], b"tl::store_global_256", "void", 2, original)
    require(ref_is(unwrap(stored_value), dest["id"]), "RMS store must write the multiplied vector")
    load_base, load_idx = vector_pointer(address_operand(load_addr), PACK)
    store_base, store_idx = vector_pointer(address_operand(store_addr), PACK)
    require(load_base["kind"] == store_base["kind"] == "DeclRefExpr"
            and slots.get(load_base["referencedDecl"]["id"]) == source
            and slots.get(store_base["referencedDecl"]["id"]) == output,
            "RMS vector load or store targets a different view")

    packs = intrinsic(children(scale)[0], b"make_ulonglong4", PACK, 4, original)
    epsilons = []
    for pack in packs:
        cast = operand(pack, "UnaryOperator", "unsigned long long", opcode="*")
        address = operand(cast, "CStyleCastExpr", "unsigned long long *", cast="BitCast")
        temporary = operand(address, "UnaryOperator", "float2 *", opcode="&")
        pair = intrinsic(temporary, b"make_float2", "float2", 2, original)
        epsilons.extend(scale_epsilon(lane, acc_id, width, original) for lane in pair)
    require(len(set(epsilons)) == 1, "RMS epsilon varies between lanes of one packed vector")

    for stmt, (outer, inner) in zip(block[3:-1], PACK_MEMBERS):
        target, product = assignment(stmt)
        product = unwrap(product)
        require(product["kind"] == "BinaryOperator" and product.get("opcode") == "*"
                and qual_type(product) == "float", "RMS output must multiply input by its computed scale")
        data_lane, scale_lane = children(product)
        packed_component(target, dest["id"], outer, inner)
        packed_component(data_lane, loaded["id"], outer, inner)
        packed_component(scale_lane, scale["id"], outer, inner)

    operands = children(square)
    beginnings = None
    for col in range(count):
        accesses = [parameter_access(arg, slots, lanes, (reduction, col)) for arg in operands]
        require(all(slot == source for slot, _ in accesses) and accesses[0][1] == accesses[1][1],
                "RMS square multiplies two different input members")
        offsets = accesses[0][1]
        if col == 0:
            beginnings = offsets
            require(all(0 <= pos < rows * width and pos % width == 0 for pos in beginnings),
                    "RMS reduction must start at the beginning of a valid row")
        require(all(pos == start + col for pos, start in zip(offsets, beginnings)),
                "RMS reduction does not follow the source row in order")

    require(sorted(beginnings[lane] for lane in active) == list(range(0, rows * width, width)),
            "RMS output lanes must own all rows exactly once")
    for tile_id in range(tiles):
        offsets_in = index_values(load_idx, lanes, (tile, tile_id))
        offsets_out = index_values(store_idx, lanes, (tile, tile_id))
        require(all(offsets_in[lane] == offsets_out[lane] == beginnings[lane] + tile_id * len(PACK_MEMBERS)
                    for lane in active), "RMS vector input and output do not match the checked reduction row")

    return {**parsed, "staged_rms": {
        "parameters": params, "shape": params[output]["shape"], "store": output,
        "expression": {"op": "row_rms", "source": source, "width": width, "epsilon": epsilons[0]},
    }}


def register(registry):
    registry.stage("cuda.normalization", "cuda_ast", "cuda_ast", capture)
