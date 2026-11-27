"""Clang-based CUDA capture and checked pointwise member computation."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
from intentlift.core.errors import require, LiftError
from ._cuda_ast import (COMPARISONS, assignment, callee, children, declaration, guard, index_values,
                       literal_integer, ordered_body, ordered_loop, qual_type, ref_is,
                       source_spelling, subscript, unwrap)
from ._cuda_math import capture_math


def parse(request, context, registry):
    path = Path(request["source"])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == request["source_sha256"],
            "source changed since Bench described it", "input_invalid")
    command = [request["clang"], "-x", "cuda", "--cuda-path=" + request["cuda_path"],
               "--cuda-gpu-arch=" + request["arch"], "--cuda-device-only", "-nocudalib",
               "-Wno-unknown-cuda-version", "-Wno-deprecated-literal-operator", "-std=c++17"]
    for directory in request["include_dirs"]:
        command += ["-I", directory]
    for header in request.get("force_includes", []):
        command += ["-include", header]
    command += ["-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang",
                "-ast-dump-filter=" + request["entry"], str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        errors = [line for line in result.stderr.splitlines() if "error:" in line]
        if errors and all("[-Waddress-of-temporary]" in line for line in errors):
            # Some nvcc snapshots use a temporary vector's address within its
            # full expression. Only retry this diagnostic; capture still checks
            # the resulting AST, and Bench independently compiles the source.
            compatible = [*command[:-1], "-Wno-address-of-temporary", command[-1]]
            result = subprocess.run(compatible, capture_output=True, text=True)
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


def scalar_literal(node, dtype):
    node = unwrap(node)
    sign = 1
    if node["kind"] == "UnaryOperator" and node.get("opcode") == "-":
        sign, node = -1, unwrap(children(node)[0])
    require(node["kind"] == "FloatingLiteral" and qual_type(node) == dtype,
            "expected a typed CUDA floating literal")
    value = sign * float(node["value"])
    require(math.isfinite(value), "expected a finite CUDA initializer")
    return value


def private_at(node, decl_id, index_id=None):
    base, index = subscript(node)
    index = unwrap(index)
    return ref_is(base, decl_id) and (
        ref_is(index, index_id) if index_id is not None
        else index["kind"] == "IntegerLiteral" and int(index["value"]) == 0
    )


def parameter_access(node, bindings, lanes, local=None, block=None):
    base, index = subscript(node)
    require(base["kind"] == "DeclRefExpr" and base["referencedDecl"]["id"] in bindings,
            "expected direct CUDA parameter access")
    return bindings[base["referencedDecl"]["id"]], index_values(index, lanes, local, block)


def capture_segmented_scan(statements, bindings, contract):
    """Lift disjoint scan fibres after checking staging and every launch owner."""
    require(len(statements) == 6 and len(bindings) == 2,
            "expected one input and one output with shared staging around an inclusive CUDA scan")
    declaration_stmt, load, before, call, after, store = statements
    shared = declaration(declaration_stmt, {"float[]"})
    require(shared.get("storageClass") == "extern" and any(
        child["kind"] == "CUDASharedAttr" for child in children(shared)), "scan requires external shared storage")
    for barrier in (before, after):
        fn, args = callee(barrier)
        require(fn["referencedDecl"]["name"] == "__syncthreads" and not args,
                "shared scan must synchronize both staging and output")
    fn, args = callee(call)
    lanes = contract["block"][0]
    segment = 32
    require(contract["block"][1:] == [1, 1] and lanes in {32, 64, 128, 256, 512, 1024}
            and 0 < math.prod(contract["grid"]) <= 1024, "unsupported scan launch geometry")
    spelling = source_spelling(fn, Path(contract["source"]).read_bytes())
    require(fn["referencedDecl"]["kind"] == "CXXMethodDecl"
            and spelling == f"tl::CumSum2D<{lanes},1,false>::run".encode()
            and len(args) == 4 and literal_integer(args[2]) == 1, "unsupported scan operator or logical row count")
    for operand in args[:2]:
        operand = unwrap(operand)
        require(operand["kind"] == "UnaryOperator" and operand.get("opcode") == "&"
                and private_at(children(operand)[0], shared["id"]), "scan must use the same shared span in place")
    width = literal_integer(args[3])
    require(0 < width <= lanes and width % segment == 0 and 4 * width <= contract["shared_mem"],
            "scan requires complete warp segments and a sufficient launch allocation")

    fibres = []
    for block in itertools.product(*(range(n) for n in contract["grid"])):
        load_lanes, loads = guard(load, lanes, block)
        store_lanes, stores = guard(store, lanes, block)
        require(load_lanes == store_lanes and len(load_lanes) == width
                and len(loads) == len(stores) == 1, "scan staging must own exactly one full fibre per block")
        staged, source = assignment(loads[0])
        target, result = assignment(stores[0])

        def shared_offsets(expression):
            base, index = subscript(expression)
            require(ref_is(base, shared["id"]), "scan stages use different allocations")
            return index_values(index, lanes, block=block)

        written, read_back = shared_offsets(staged), shared_offsets(result)
        require([written[i] for i in load_lanes] == [read_back[i] for i in store_lanes] == list(range(width)),
                "staging changes the shared scan coordinate")
        src, in_offsets = parameter_access(source, bindings, lanes, block=block)
        dest, out_offsets = parameter_access(target, bindings, lanes, block=block)
        require(src["mode"] == "input" and dest["mode"] == "output"
                and src["dtype"] == dest["dtype"] == "f32" and src["shape"] == dest["shape"],
                "scan requires corresponding read-only and output f32 views")
        in_order = tuple(in_offsets[i] for i in load_lanes)
        out_order = tuple(out_offsets[i] for i in store_lanes)
        require(in_order == out_order, "scan output does not correspond to the input fibre")
        fibres.append(out_order)
    shape = dest["shape"]
    extent = math.prod(shape)
    require(extent == len(fibres) * width and sorted(itertools.chain.from_iterable(fibres)) == list(range(extent)),
            "scan ownership must cover every member without overlap")
    for axis, dimension in enumerate(shape):
        if dimension != width:
            continue
        stride = math.prod(shape[axis + 1:])
        if all(row[0] // stride % width == 0
                and all(address == row[0] + i * stride for i, address in enumerate(row)) for row in fibres):
            return {"parameters": contract["parameters"], "shape": shape, "store": dest["slot"],
                    "expression": {"op": "segmented_scan", "source": src["slot"], "axis": axis,
                                   "segment": segment}}
    raise LiftError("capability_missing", "scan fibres do not map to a full logical axis")


def row_update(stmt, acc):
    dest, update = assignment(stmt)
    require(private_at(dest, acc), "expected an ordered update of the accumulator cell")
    return unwrap(update)


def capture_row(stmt, bindings, contract):
    shared_mode = len(stmt) == 7
    if shared_mode:
        shared_decl, acc_decl, init, copy, barrier, compute, store = stmt
        require(contract["shared_mem"] > 0, "GEMV staging needs trusted shared-memory allocation")
    else:
        acc_decl, init, compute, store = stmt
        require(contract["shared_mem"] == 0, "direct reduction does not use shared memory")
    acc = declaration(acc_decl, {"float[1]"})
    require(not children(acc) and not acc.get("storageClass"), "expected an automatic private accumulator")
    acc = acc["id"]
    dest, value = assignment(init)
    require(private_at(dest, acc), "expected initialization of the private accumulator")
    initial = scalar_literal(value, "float")
    lanes = contract["block"][0]
    require(contract["grid"] == [1, 1, 1] and contract["block"][1:] == [1, 1] and 0 < lanes <= 1024,
            "row owner capture requires one one-dimensional block")
    active, stores = guard(store, lanes)
    require(len(stores) == 1, "row owner needs exactly one output store")
    dest, value = assignment(stores[0])
    require(private_at(value, acc), "row output must store the computed accumulator")
    output, output_indices = parameter_access(dest, bindings, lanes)
    rows = output["shape"][0]
    row_by_lane = tuple(output_indices[lane] for lane in active)
    require(output["mode"] == "output" and output["dtype"] == "f32" and len(output["shape"]) == 1
            and sorted(row_by_lane) == list(range(rows)), "row stores must cover the output exactly once")

    iv, limit, update = ordered_loop(compute)
    update = row_update(update, acc)
    if update["kind"] == "BinaryOperator" and update.get("opcode") == "+":
        operands, combine = children(update), "sum"
    else:
        function, operands = callee(update)
        require(function["referencedDecl"]["name"] == "max" and qual_type(update) == "float",
                "expected f32 maximum-number reduction")
        combine = "maximum"
    require(len(operands) == 2 and private_at(operands[0], acc) and qual_type(update) == "float",
            "expected f32 accumulator as the first operand")
    contribution = unwrap(operands[1])

    def factor(node, parameter_dtype, matrix):
        info, _ = parameter_access(node, bindings, lanes, (iv, 0))
        expected_shape = [rows, limit] if matrix else [limit]
        require(info["mode"] == "input" and info["dtype"] == parameter_dtype and info["shape"] == expected_shape,
                "ordered row factor ABI differs from its index mapping")
        for k in range(limit):
            _, indices = parameter_access(node, bindings, lanes, (iv, k))
            require(all(0 <= item < math.prod(expected_shape) for item in indices)
                    and all(indices[lane] == (row * limit if matrix else 0) + k
                        for lane, row in zip(active, row_by_lane)), "row factor has a noncanonical index")
        return info["slot"]

    if not shared_mode:
        if combine == "sum" and contribution["kind"] == "BinaryOperator" and contribution.get("opcode") == "*":
            factors = []
            for operand in children(contribution):
                info, _ = parameter_access(operand, bindings, lanes, (iv, 0))
                matrix = len(info["shape"]) == 2
                factors.append({"slot": factor(operand, "f32", matrix), "matrix": matrix})
        else:
            factors = [{"slot": factor(contribution, "f32", True), "matrix": True}]
    else:
        require(combine == "sum" and contribution["kind"] == "BinaryOperator"
                and contribution.get("opcode") == "*", "expected a widened shared-memory dot product")
        shared = declaration(shared_decl, {"unsigned char[]"})
        require(shared.get("storageClass") == "extern" and any(
            child["kind"] == "CUDASharedAttr" for child in children(shared)), "expected external shared storage")
        barrier_function, args = callee(barrier)
        require(barrier_function["referencedDecl"]["name"] == "__syncthreads" and not args,
                "shared copy must be followed by a block barrier")

        def shared_indices(node, local):
            base, index = subscript(node)
            require(base["kind"] == "CStyleCastExpr" and base.get("castKind") == "BitCast"
                    and qual_type(base) in {"half_t *", "cutlass::half_t *"}
                    and ref_is(unwrap(children(base)[0]), shared["id"]), "expected aligned shared f16 access")
            result = index_values(index, lanes, local)
            require(all(0 <= item and 2 * (item + 1) <= contract["shared_mem"] for item in result),
                    "shared f16 index exceeds launch allocation")
            return result

        copy_iv, copy_extent, copies = ordered_body(copy)
        require(copy_extent == limit and len(copies) == 2, "expected two complete operand copies")
        staged = {}
        for copy_stmt in copies:
            require(copy_stmt["kind"] == "CXXOperatorCallExpr" and len(children(copy_stmt)) == 3,
                    "expected one shared half assignment")
            func, dest, source = children(copy_stmt)
            require(func["kind"] == "ImplicitCastExpr"
                    and children(func)[0].get("referencedDecl", {}).get("name") == "operator="
                    and source["type"].get("desugaredQualType", qual_type(source)) == "const cutlass::half_t",
                    "expected a CUTLASS half copy")
            for k in range(copy_extent):
                local = (copy_iv, k)
                addresses = shared_indices(dest, local)
                info, src_indices = parameter_access(source, bindings, lanes, local)
                require(info["mode"] == "input" and info["dtype"] == "f16"
                        and all(0 <= index < math.prod(info["shape"]) for index in src_indices),
                        "invalid shared copy source")
                for addr, src in zip(addresses, src_indices):
                    identity = (info["slot"], src, k)
                    if addr in staged:
                        require(lanes <= 32 and staged[addr] == identity, "ambiguous shared-memory write")
                    staged[addr] = identity
        abi = contract["parameters"]
        factors = []
        for term in children(contribution):
            read = half_to_float_operand(term)
            addresses = shared_indices(read, (iv, 0))
            require(addresses[active[0]] in staged, "shared operand is uninitialized")
            slot = staged[addresses[active[0]]][0]
            shape = abi[slot]["shape"]
            matrix = shape == [rows, limit]
            require(matrix or shape == [limit], "shared row factor shape mismatch")
            for k in range(limit):
                addresses = shared_indices(read, (iv, k))
                require(all(address in staged for address in addresses)
                        and all(staged.get(addresses[lane]) == (slot, (row * limit if matrix else 0) + k, k)
                            for lane, row in zip(active, row_by_lane)), "shared read does not match the staged factor")
            factors.append({"slot": slot, "matrix": matrix, "widen": True})
        require(sum(f["matrix"] for f in factors) == 1, "expected one vector and one matrix factor")

    return {"parameters": contract["parameters"], "shape": output["shape"], "store": output["slot"],
            "expression": {"op": "ordered_row", "initial": initial, "combine": combine, "factors": factors}}



def unwrapped_call(node):
    while node["kind"] in {"ExprWithCleanups", "MaterializeTemporaryExpr"}:
        require(len(children(node)) == 1, "unexpected C++ temporary")
        node = children(node)[0]
    return unwrap(node)


def overloaded_assignment(node):
    node = unwrapped_call(node)
    require(node["kind"] == "CXXOperatorCallExpr" and len(children(node)) == 3,
            "expected vector assignment")
    func, dest, value = children(node)
    require(func["kind"] == "ImplicitCastExpr"
            and children(func)[0].get("referencedDecl", {}).get("name") == "operator=",
            "unexpected CUDA vector operation")
    return dest, value


def vector_pointer(node, dtype):
    node = unwrap(node)
    require(node["kind"] == "UnaryOperator" and node.get("opcode") == "*" and qual_type(node) in {dtype, "const " + dtype},
            "expected an aligned CUDA vector access")
    cast = unwrap(children(node)[0])
    require(cast["kind"] == "CStyleCastExpr" and cast.get("castKind") == "BitCast"
            and qual_type(cast) == dtype + " *", "expected a CUDA vector pointer cast")
    offset = unwrap(children(cast)[0])
    require(offset["kind"] == "BinaryOperator" and offset.get("opcode") == "+",
            "expected an explicit vector pointer offset")
    base, index = children(offset)
    return unwrap(base), index


def integral_literal(node):
    node = unwrap(node) if node["kind"] != "ImplicitCastExpr" or node.get("castKind") != "IntegralCast" else children(node)[0]
    if node["kind"] == "UnaryOperator" and node.get("opcode") == "-":
        return -integral_literal(children(node)[0])
    require(node["kind"] == "IntegerLiteral", "expected an integral CUDA constant")
    return int(node["value"])


def negative_infinity(node):
    node = unwrap(node)
    require(node["kind"] == "UnaryOperator" and node.get("opcode") == "-", "expected negative infinity")
    function, args = callee(unwrap(children(node)[0]))
    require(function["referencedDecl"]["name"] == "__int_as_float" and len(args) == 1
            and integral_literal(args[0]) == 0x7f800000, "expected CUDA negative-infinity initializer")


def selection_phase(loop, peak, scores, destination, dtype, size):
    iv, limit, body = ordered_body(loop)
    require(limit == size and len(body) == 3 and body[1]["kind"] == "IfStmt"
            and len(children(body[1])) == 3, "expected per-candidate selection")
    local = declaration(body[0], {dtype})
    require(not children(local) and not local.get("storageClass"), "unexpected selection temporary")
    condition, yes, no = children(body[1])
    condition = unwrap(condition)
    require(condition["kind"] == "BinaryOperator" and condition.get("opcode") == "=="
            and private_at(children(condition)[0], peak) and private_at(children(condition)[1], scores, iv),
            "selection predicate must compare score against its row maximum")
    values = []
    for branch in (yes, no):
        require(branch["kind"] == "CompoundStmt" and len(children(branch)) == 1,
                "selection branch must assign one value")
        target, value = assignment(children(branch)[0])
        require(ref_is(unwrap(target), local["id"]), "selection must assign its local")
        values.append(value)
    target, value = assignment(body[2])
    require(private_at(values[1], destination, iv) and private_at(target, destination, iv)
            and ref_is(unwrap(value), local["id"]), "selection must preserve an unselected candidate")
    return iv, values[0]


def capture_indexed_selection(statements, bindings, contract):
    params = contract["parameters"]
    modes = sorted((p["dtype"], p["mode"]) for p in params)
    require(modes == [("f32", "input"), ("f32", "output"), ("i32", "output")],
            "unsupported paired selection ABI")
    require(len(statements) == 6, "unsupported paired selection body")
    scores_decl, indices_decl, peak_decl, index_decl, load, loop = statements
    scores = declaration(scores_decl, {"float[4]"})
    indices = declaration(indices_decl, {"int[4]"})
    peak = declaration(peak_decl, {"float[1]"})
    chosen = declaration(index_decl, {"int[1]"})
    require(all(not children(v) and not v.get("storageClass") for v in (scores, indices, peak, chosen)),
            "selection state must be made of automatic private cells")
    scores, indices, peak, chosen = (v["id"] for v in (scores, indices, peak, chosen))

    fragment = 4
    dest, value = overloaded_assignment(load)
    dst_base, dst_idx = vector_pointer(dest, "float4")
    src_base, src_idx = vector_pointer(value, "float4")
    require(ref_is(dst_base, scores) and integral_literal(dst_idx) == 0
            and src_base["kind"] == "DeclRefExpr", "expected contiguous score fragments")
    logits = bindings.get(src_base["referencedDecl"]["id"])
    require(logits is not None and logits["dtype"] == "f32" and logits["mode"] == "input"
            and len(logits["shape"]) == 2, "selection must load its score matrix input")
    rows, width = logits["shape"]
    cohort = width // fragment
    lanes = contract["block"][0]
    require(width == cohort * fragment and cohort in {2, 4, 8, 16, 32}
            and lanes == rows * cohort and lanes % 32 == 0 and lanes <= 1024
            and contract["grid"] == [1, 1, 1] and contract["block"][1:] == [1, 1]
            and contract["shared_mem"] == 0, "selection requires contiguous full shuffle groups")
    require(index_values(src_idx, lanes) == tuple(lane * fragment for lane in range(lanes)),
            "score fragment mapping is not row-major")

    step, picks, rounds = ordered_body(loop)
    require(len(rounds) == 11 and picks <= width, "unsupported selection recurrence")
    reset, fill, peak_init, peak_local, peak_warp, match, index_init, index_local, index_warp, mask, stores = rounds
    reset = declaration(reset, {"int"})
    require(not reset.get("storageClass") and len(children(reset)) == 1, "invalid candidate index initializer")
    empty = integral_literal(children(reset)[0])
    require(-(1 << 31) < empty < 0, "unmatched candidate index must be negative")
    dest, value = overloaded_assignment(fill)
    base, offset = vector_pointer(dest, "int4")
    function, values = callee(unwrapped_call(value))
    require(ref_is(base, indices) and integral_literal(offset) == 0
            and function["referencedDecl"]["name"] == "make_int4" and len(values) == fragment
            and all(ref_is(unwrap(arg), reset["id"]) for arg in values), "invalid candidate initialization")

    target, initial = assignment(peak_init)
    require(private_at(target, peak), "unexpected score initializer")
    negative_infinity(initial)
    target, initial = assignment(index_init)
    require(private_at(target, chosen) and integral_literal(initial) == -(1 << 31),
            "invalid integer max initializer")

    def local_max(phase, accumulator, source, dtype):
        index, count, update = ordered_loop(phase)
        function, args = callee(row_update(update, accumulator))
        require(count == fragment and function["referencedDecl"]["name"] == "max"
                and len(args) == 2 and qual_type(args[0]) == dtype
                and private_at(args[0], accumulator) and private_at(args[1], source, index),
                "unexpected fragment maximum")

    local_max(peak_local, peak, scores, "float")
    local_max(index_local, chosen, indices, "int")

    original = Path(contract["source"]).read_bytes()

    def warp_max(phase, accumulator, dtype):
        function, args = callee(row_update(phase, accumulator))
        expected = f"tl::AllReduce<tl::MaxOp,{cohort},1,0,tl::NamedBarrier<{lanes}>>::run"
        begin, end = (function["range"][key] for key in ("begin", "end"))
        # Clang exposes the enclosing template arguments in the qualified token
        # range, rather than as children of the resolved DeclRefExpr.
        spelling = original[begin.get("offset", 0):end.get("offset", 0) + end["tokLen"]]
        require(b"".join(spelling.split()) == expected.encode()
                and function["referencedDecl"]["kind"] == "CXXMethodDecl"
                and len(args) == 2 and qual_type(args[0]) == dtype
                and private_at(args[0], accumulator) and args[1]["kind"] == "CXXDefaultArgExpr",
                "unexpected CUDA shuffle reduction declaration or group size")

    warp_max(peak_warp, peak, "float")
    warp_max(index_warp, chosen, "int")
    candidate_iv, candidate = selection_phase(match, peak, scores, indices, "int", fragment)
    for j in range(fragment):
        require(index_values(candidate, lanes, (candidate_iv, j))
                == tuple((lane % cohort) * fragment + j for lane in range(lanes)),
                "selection candidate index does not correspond to its score")
    _, masked = selection_phase(mask, peak, scores, scores, "float", fragment)
    sentinel = scalar_literal(masked, "float")
    active, stores = guard(stores, lanes)
    require(active == tuple(range(0, lanes, cohort)) and len(stores) == 2,
            "selection output guard must select one owner per row")
    outputs = {}
    for statement in stores:
        dst, src = assignment(statement)
        store, _ = parameter_access(dst, bindings, lanes, (step, 0))
        dtype = store["dtype"]
        require(store["mode"] == "output" and dtype in {"f32", "i32"} and dtype not in outputs
                and store["shape"] == [rows, picks] and private_at(src, peak if dtype == "f32" else chosen),
                "selection output type, shape or value mismatch")
        outputs[dtype] = store["slot"]
        for k in range(picks):
            _, offsets = parameter_access(dst, bindings, lanes, (step, k))
            require(all(offsets[lane] == row * picks + k for row, lane in enumerate(active)),
                    "selection output coordinates do not cover the complete result")
    return {"parameters": params, "shape": [rows, picks], "store": outputs["f32"],
            "expression": {"op": "indexed_selection", "input": logits["slot"], "indices": outputs["i32"],
                           "sentinel": sentinel, "empty_index": empty}}


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
    if "staged_attention" in parsed:
        return parsed["staged_attention"]
    if "staged_convolution" in parsed:
        return parsed["staged_convolution"]
    fn, contract = parsed["function"], parsed["contract"]
    parameters = [p for p in children(fn) if p["kind"] == "ParmVarDecl"]
    abi = contract["parameters"]
    require(len(parameters) == len(abi), "CUDA ABI does not match declared parameters", "input_invalid")
    body = next(p for p in children(fn) if p["kind"] == "CompoundStmt")
    statements = children(body)
    scan = False
    if len(statements) == 6 and statements[0]["kind"] == "DeclStmt":
        local = children(statements[0])
        scan = len(local) == 1 and qual_type(local[0]) == "float[]" and statements[3]["kind"] == "CallExpr"
    bindings = {}
    for i, (p, info) in enumerate(zip(parameters, abi)):
        ty = qual_type(p).replace("__restrict", "").strip()
        pointer_types = {"float *": "f32", "half_t *": "f16", "cutlass::half_t *": "f16", "int *": "i32"}
        require(pointer_types.get(ty.removeprefix("const ")) == info["dtype"],
                "CUDA pointer type differs from supported storage type", dtype=info["dtype"])
        require(len(info["shape"]) in ({1, 2, 3, 4} if scan else {1, 2})
                and all(type(dim) is int and dim > 0 for dim in info["shape"]),
                "capture requires positive static rank-one or rank-two views")
        require(info["name"] == p["name"], "CUDA parameter order/name mismatch", "input_invalid")
        require(info["mode"] in {"input", "output", "inout"}, "unsupported ABI effect")
        require(not (ty.startswith("const ") and info["mode"] != "input"), "const parameter cannot be output", "input_invalid")
        require(info["dtype"] != "f16" or info["mode"] == "input", "binary16 support requires read-only views")
        bindings[p["id"]] = {**info, "slot": i}
    outputs = [i for i, p in enumerate(abi) if p["mode"] != "input"]
    if scan:
        return capture_segmented_scan(statements, bindings, contract)
    if len(outputs) == 2:
        return capture_indexed_selection(statements, bindings, contract)
    require(len(outputs) == 1, "pointwise capture requires one observable view")
    shape = tuple(abi[outputs[0]]["shape"])
    if len(shape) == 1 and (
        (len(statements) == 4 and statements[2]["kind"] == "ForStmt")
        or (len(statements) == 7 and statements[3]["kind"] == statements[5]["kind"] == "ForStmt")
    ):
        return capture_row(statements, bindings, contract)
    n = math.prod(shape)
    require(0 < n <= 1024 and contract["grid"] == [1, 1, 1]
            and contract["block"] == [n, 1, 1] and contract["shared_mem"] == 0,
            "threadIdx.x member map requires one block covering the complete extent")
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
        if kind == "CallExpr":
            return capture_math(node, expression)
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
