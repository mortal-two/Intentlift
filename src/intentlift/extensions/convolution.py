"""Lift a shared-staged 2D stencil after proving each operand's provenance."""
import itertools
import math
import operator
from pathlib import Path
import re

from intentlift.core.errors import require
from ._cuda_ast import (
    assignment, callee, children, declaration, ordered_body, ordered_loop,
    qual_type, ref_is, source_spelling, subscript, unwrap, walk,
)
from .cuda import half_to_float_operand, overloaded_assignment, scalar_literal, unwrapped_call, vector_pointer


BINARY = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.floordiv,
          "%": operator.mod, "&": operator.and_, ">>": operator.rshift,
          "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
          "==": operator.eq, "!=": operator.ne}


def index(node):
    """Compile a checked launch-index expression to an integer evaluator."""
    kind, dtype = node["kind"], qual_type(node)
    require(dtype in {"int", "unsigned int", "bool"}, "unsupported stencil index type")
    if kind in {"ImplicitCastExpr", "CStyleCastExpr", "ParenExpr"}:
        require(kind == "ParenExpr" or node.get("castKind") in {"NoOp", "LValueToRValue", "IntegralCast"},
                "unsupported stencil index cast")
        inner = index(children(node)[0])
        return (lambda env: inner(env) % (1 << 32)) if dtype == "unsigned int" else inner
    if kind == "IntegerLiteral":
        value = int(node["value"])
        return lambda env: value
    if kind == "DeclRefExpr":
        name = node["referencedDecl"]["id"]

        def local(env):
            require(name in env, "unbound stencil index")
            return env[name]
        return local
    if kind == "PseudoObjectExpr":
        refs = {(c["referencedDecl"]["name"], qual_type(c)) for c in walk(node) if c.get("kind") == "DeclRefExpr"}
        members = {c["name"] for c in walk(node) if c.get("kind") == "MemberExpr"}
        if refs == {("threadIdx", "const __cuda_builtin_threadIdx_t")} and members == {"__fetch_builtin_x"}:
            return lambda env: env["lane"]
        require(refs == {("blockIdx", "const __cuda_builtin_blockIdx_t")} and len(members) == 1,
                "unsupported stencil launch index")
        member = next(iter(members))
        require(member in {"__fetch_builtin_x", "__fetch_builtin_y", "__fetch_builtin_z"},
                "unsupported stencil block axis")
        axis = "xyz".index(member[-1])
        return lambda env: env["block"][axis]
    require(kind == "BinaryOperator" and node["opcode"] in {*BINARY, "&&", "||"},
            "unsupported stencil index operation")
    op = node["opcode"]
    left, right = (index(child) for child in children(node))
    if op == "&&":
        return lambda env: left(env) and right(env)
    if op == "||":
        return lambda env: left(env) or right(env)

    def evaluate(env):
        a, b = left(env), right(env)
        if op in {"/", "%"}:
            require(a >= 0 and b > 0, "unsupported signed division in stencil mapping")
        if op == ">>":
            require(a >= 0 and 0 <= b < 32, "invalid shift in stencil mapping")
        value = BINARY[op](a, b)
        if dtype == "unsigned int":
            return value % (1 << 32)
        require(-(1 << 31) <= value < (1 << 31), "signed overflow in stencil mapping")
        return value
    return evaluate


def direct_access(node):
    base, offset = subscript(node)
    require(base["kind"] == "DeclRefExpr", "expected a direct stencil view or local array")
    return base["referencedDecl"]["id"], index(offset)


def address_operand(node):
    while node["kind"] in {"ImplicitCastExpr", "ParenExpr"}:
        require(node["kind"] == "ParenExpr" or node.get("castKind") in {"NoOp", "BitCast"},
                "unsupported async copy address cast")
        node = children(node)[0]
    require(node["kind"] == "UnaryOperator" and node.get("opcode") == "&", "expected address of a staged member")
    return children(node)[0]


def half_constructor(node):
    while node["kind"] in {"ParenExpr", "ImplicitCastExpr", "ExprWithCleanups", "MaterializeTemporaryExpr",
                           "CXXFunctionalCastExpr", "CStyleCastExpr"}:
        require(len(children(node)) == 1, "unsupported half constructor wrapper")
        node = children(node)[0]
    require(node["kind"] == "CXXConstructExpr" and len(children(node)) == 1
            and node["type"].get("desugaredQualType", qual_type(node)) == "cutlass::half_t"
            and qual_type(children(node)[0]) == "float", "expected an f32 to CUTLASS binary16 conversion")
    return children(node)[0]


def unpack(flat, shape):
    result = []
    for extent in reversed(shape):
        flat, value = divmod(flat, extent)
        result.append(value)
    require(flat == 0, "stencil coordinate exceeds its view")
    return tuple(reversed(result))


def capture(parsed, context, registry):
    fn, contract = parsed["function"], parsed["contract"]
    body = next(n for n in children(fn) if n["kind"] == "CompoundStmt")
    statements = children(body)
    # Distinguish this schedule by shared conditional staging and one output
    # fragment. Rank alone does not select a convolution.
    if not any(n.get("kind") == "DeclRefExpr" and
               n.get("referencedDecl", {}).get("name") == "cp_async_gs_conditional" for n in walk(body)):
        return parsed
    if len(statements) != 11:
        return parsed
    abi = contract["parameters"]
    if sorted(p["mode"] for p in abi) != ["input", "input", "output"] or {p["dtype"] for p in abi} != {"f16"}:
        return parsed
    require(all(len(p["shape"]) == 4 for p in abi), "stencil expects rank-four views")
    decls = [p for p in children(fn) if p["kind"] == "ParmVarDecl"]
    require(len(decls) == len(abi) and all(p["name"] == d["name"]
            and qual_type(d).replace("__restrict", "").strip() ==
            ("half_t *" if p["mode"] == "output" else "const half_t *")
            for p, d in zip(abi, decls)), "stencil CUDA ABI mismatch")
    bindings = {d["id"]: p for d, p in zip(decls, abi)}
    shared_decl, acc_decl, init, initial_data, initial_filter, commit, pipeline, wait, sync, final, store = statements
    shared = declaration(shared_decl, {"unsigned char[]"})
    require(shared.get("storageClass") == "extern" and any(
        child["kind"] == "CUDASharedAttr" for child in children(shared)), "expected shared stencil staging")
    scratch = contract["shared_mem"] // 2
    require(scratch > 0, "missing stencil shared allocation")
    original = Path(contract["source"]).read_bytes()

    def helper(stmt, name):
        function, arguments = callee(stmt)
        require(source_spelling(function, original) == name and not arguments,
                "stencil async or block synchronization does not match")
    helper(commit, b"tl::cp_async_commit")
    helper(wait, b"tl::cp_async_wait<0>")
    helper(sync, b"__syncthreads")

    def shared_access(node):
        base, offset = subscript(node)
        require(base["kind"] == "CStyleCastExpr" and base.get("castKind") == "BitCast"
                and qual_type(base) in {"half_t *", "cutlass::half_t *"}
                and ref_is(unwrap(children(base)[0]), shared["id"]), "expected binary16 access into shared staging")
        return index(offset)

    def valid_input(ident):
        view = bindings.get(ident)
        require(view is not None and view["mode"] == "input", "stencil source must be an input view")
        return view

    def write(staged, address, value):
        require(0 <= address < scratch and address not in staged,
                "shared stencil write exceeds allocation or overlaps another write")
        staged[address] = value

    def copy_loop(stmt):
        iv, count, parts = ordered_body(stmt)
        locals_ = []
        for local in parts[:-1]:
            value = declaration(local, {"bool"})
            require(len(children(value)) == 1 and not value.get("storageClass"), "expected initialized copy predicate")
            locals_.append((value["id"], index(children(value)[0])))
        function, args = callee(parts[-1])
        match = re.fullmatch(rb"tl::cp_async_gs(_conditional)?<(4|8|16)>", source_spelling(function, original))
        require(match is not None, "unrecognized shared memory copy")
        conditional, width = bool(match[1]), int(match[2]) // 2
        require(len(args) == 2 + int(conditional), "unexpected async copy parameters")
        dst = shared_access(address_operand(args[0]))
        src, offset = direct_access(address_operand(args[1]))
        view = valid_input(src)
        pred = index(args[2]) if conditional else lambda env: True

        def run(lane_env, staged):
            env = lane_env.copy()
            for iteration in range(count):
                env[iv] = iteration
                for name, value in locals_:
                    env[name] = value(env)
                address = dst(env)
                require(address % width == 0, "misaligned shared async copy")
                if pred(env):
                    start = offset(env)
                    require(start % width == 0 and 0 <= start and start + width <= math.prod(view["shape"]),
                            "async copy reads outside a stencil view")
                    for k in range(width):
                        write(staged, address + k, (view["position"], start + k))
                else:
                    for k in range(width):
                        write(staged, address + k, None)
        return view, run

    def padded_loop(stmt):
        outer, outer_n, parts = ordered_body(stmt)
        require(len(parts) == 1, "expected two-level guarded staging")
        inner, inner_n, parts = ordered_body(parts[0])
        require(len(parts) == 4, "expected one guarded half assignment")
        pred_decl, half_decl, branch, stored = parts
        pred_decl = declaration(pred_decl, {"bool"})
        half_decl = declaration(half_decl, {"half_t"})
        require(len(children(pred_decl)) == 1 and not pred_decl.get("storageClass")
                and not half_decl.get("storageClass") and len(children(half_decl)) == 1
                and children(half_decl)[0]["kind"] == "CXXConstructExpr"
                and not children(children(half_decl)[0]), "unexpected staging locals")
        pred_init = index(children(pred_decl)[0])
        require(branch["kind"] == "IfStmt" and len(children(branch)) == 3, "padded load needs two branches")
        predicate, yes, no = children(branch)
        pred = index(predicate)

        def branch_assignment(block):
            require(block["kind"] == "CompoundStmt" and len(children(block)) == 1,
                    "unexpected padded load branch")
            dst, src = overloaded_assignment(children(block)[0])
            require(ref_is(unwrap(dst), half_decl["id"]), "branches must assign the staged half")
            return src

        src, src_offset = direct_access(branch_assignment(yes))
        view = valid_input(src)
        require(scalar_literal(half_constructor(branch_assignment(no)), "float") == 0.0,
                "invalid stencil padding value")
        dst, val = overloaded_assignment(stored)
        require(ref_is(unwrap(val), half_decl["id"]), "shared store must use the selected source")
        address = shared_access(dst)

        def run(lane_env, staged):
            env = lane_env.copy()
            for outer_index, inner_index in itertools.product(range(outer_n), range(inner_n)):
                env.update({outer: outer_index, inner: inner_index})
                env[pred_decl["id"]] = pred_init(env)
                if pred(env):
                    flat = src_offset(env)
                    require(0 <= flat < math.prod(view["shape"]), "guarded load reads outside the input")
                    source = view["position"], flat
                else:
                    source = None
                write(staged, address(env), source)
        return view, run

    data, first_a = copy_loop(initial_data)
    filt, first_b = copy_loop(initial_filter)
    tick, ticks, phases = ordered_body(pipeline)
    require(len(phases) == 7, "unrecognized stencil pipeline phases")
    helper(phases[0], b"tl::cp_async_wait<0>")
    helper(phases[1], b"__syncthreads")
    helper(phases[3], b"__syncthreads")
    helper(phases[-1], b"tl::cp_async_commit")
    later_data, next_a = padded_loop(phases[4])
    later_filt, next_b = copy_loop(phases[5])
    require(data == later_data and filt == later_filt and data != filt,
            "staging switches input roles between tiles")

    acc = declaration(acc_decl, {f"float[{n}]" for n in range(4, 129, 4)})
    require(not children(acc) and not acc.get("storageClass"), "stencil accumulator must be private")
    pixels = int(qual_type(acc)[6:-1])
    init_iv, init_n, statements = ordered_body(init)
    require(init_n * 4 == pixels and len(statements) == 2, "stencil accumulator is not fully initialized")
    fill = declaration(statements[0], {"float"})
    require(len(children(fill)) == 1 and not fill.get("storageClass")
            and scalar_literal(children(fill)[0], "float") == 0.0, "stencil initializer must be f32 zero")
    dst, value = overloaded_assignment(statements[1])
    dst_base, dst_index = vector_pointer(dst, "float4")
    ctor, arguments = callee(unwrapped_call(value))
    require(ref_is(dst_base, acc["id"]) and ctor["referencedDecl"]["name"] == "make_float4"
            and len(arguments) == 4 and all(ref_is(unwrap(arg), fill["id"]) for arg in arguments)
            and all(index(dst_index)({init_iv: k}) == k * 4 for k in range(init_n)),
            "stencil initializer must cover each private accumulator")

    def compute(loop):
        row, rows, parts = ordered_body(loop)
        require(rows == pixels and len(parts) == 1, "unexpected stencil fragment loop")
        k, length, update = ordered_loop(parts[0])
        target, expr = assignment(update)
        target_ref, target_offset = direct_access(target)
        expr = unwrap(expr)
        require(target_ref == acc["id"] and expr["kind"] == "BinaryOperator"
                and expr.get("opcode") == "+" and qual_type(expr) == "float", "expected ordered f32 accumulation")
        prior, contribution = children(expr)
        prior_ref, prior_offset = direct_access(prior)
        contribution = unwrap(contribution)
        require(prior_ref == acc["id"] and contribution["kind"] == "BinaryOperator"
                and contribution.get("opcode") == "*" and qual_type(contribution) == "float",
                "expected separately rounded f32 product")
        a, b = (shared_access(half_to_float_operand(term)) for term in children(contribution))
        return row, k, length, target_offset, prior_offset, a, b

    first_compute, last_compute = compute(phases[2]), compute(final)
    segment = first_compute[2]
    require(last_compute[2] == segment, "reduction tile width changes")
    out_iv, out_n, store_stmt = ordered_loop(store)
    dest, value = overloaded_assignment(store_stmt)
    dst_ref, dst_index = direct_access(dest)
    acc_ref, acc_index = direct_access(half_constructor(value))
    out = bindings.get(dst_ref)
    require(out is not None and out["mode"] == "output" and out_n == pixels and acc_ref == acc["id"],
            "stencil output must store every private value through binary16 narrowing")

    batch, height, width, channels = data["shape"]
    kh, kw, filter_channels, filters = filt["shape"]
    require(out["shape"] == [batch, height, width, filters] and filter_channels == channels
            and kh % 2 == kw % 2 == 1, "expected a same-size odd-filter NHWC stencil")
    require((ticks + 1) * segment == math.ceil(kh * kw * channels / segment) * segment,
            "CUDA pipeline length differs from the padded reduction")
    lanes = contract["block"][0]
    require(contract["block"][1:] == [1, 1] and 0 < lanes <= 128
            and math.prod(contract["grid"]) * lanes * pixels * segment * (ticks + 1) <= 1_000_000,
            "stencil launch exceeds the checked proof domain")
    padding_h, padding_w = kh // 2, kw // 2

    owners = set()
    for block in itertools.product(*(range(count) for count in contract["grid"])):
        local = [{"block": block, "lane": lane} for lane in range(lanes)]
        staging = []
        for tile in range(ticks + 1):
            staged = {}
            for env in local:
                if tile == 0:
                    first_a(env, staged)
                    first_b(env, staged)
                else:
                    phase_env = {**env, tick: tile - 1}
                    next_a(phase_env, staged)
                    next_b(phase_env, staged)
            staging.append(staged)
        for env in local:
            for pixel in range(pixels):
                output_env = {**env, out_iv: pixel}
                flat = dst_index(output_env)
                require(0 <= flat < math.prod(out["shape"]) and flat not in owners and acc_index(output_env) == pixel,
                        "stencil stores are not a unique full output map")
                owners.add(flat)
                batch_idx, y, x, feature = unpack(flat, out["shape"])
                for tile, staged in enumerate(staging):
                    row, k, _, out_cell, in_cell, lhs, rhs = last_compute if tile == ticks else first_compute
                    for term in range(segment):
                        calc_env = {**env, row: pixel, k: term}
                        require(out_cell(calc_env) == in_cell(calc_env) == pixel,
                                "stencil accumulates into a different private coordinate")
                        li, ri = lhs(calc_env), rhs(calc_env)
                        require(li in staged and ri in staged, "stencil computation reads an unstaged element")
                        h, rest = divmod(tile * segment + term, kw * channels)
                        w, c = divmod(rest, channels)
                        iy, ix = y + h - padding_h, x + w - padding_w
                        input_value = (data["position"], ((batch_idx * height + iy) * width + ix) * channels + c) \
                            if 0 <= iy < height and 0 <= ix < width else None
                        filter_value = (filt["position"], ((h * kw + w) * channels + c) * filters + feature) \
                            if h < kh else None
                        require(staged[li] == input_value and staged[ri] == filter_value,
                                "staged operands differ from the ordered padded convolution")
    require(len(owners) == math.prod(out["shape"]), "stencil launch leaves output members unwritten")
    return {**parsed, "staged_convolution": {
        "parameters": abi, "shape": out["shape"], "store": out["position"],
        "expression": {"op": "staged_convolution", "data": data["position"], "filter": filt["position"],
                       "terms": (ticks + 1) * segment, "padding": (padding_h, padding_w)},
    }}


def register(registry):
    registry.stage("cuda.convolution", "cuda_ast", "cuda_ast", capture)
