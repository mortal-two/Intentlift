"""Checked lifting of a single-tile staged attention schedule."""
import itertools
import math
from pathlib import Path

from intentlift.core.errors import LiftError, require


def children(node):
    return node.get("inner", [])


def dtype(node):
    return node.get("type", {}).get("qualType")


def walk(node):
    yield node
    for item in children(node):
        yield from walk(item)


class Schedule:
    """Canonical AST projection with lexical declaration identities."""
    def __init__(self, identities, source):
        self.identities = identities
        self.source = source
        self.loop_depth = 0

    def node(self, node):
        kind = node["kind"]
        parts = children(node)
        def arg(i):
            return self.node(parts[i])

        if kind in {"ParenExpr", "ExprWithCleanups", "MaterializeTemporaryExpr"}:
            return arg(0)
        if kind == "ImplicitCastExpr":
            require(node.get("castKind") in {"NoOp", "LValueToRValue", "ArrayToPointerDecay",
                    "FunctionToPointerDecay", "BuiltinFnToFnPtr", "IntegralCast", "UserDefinedConversion", "BitCast"},
                    "unhandled implicit cast in attention schedule", cast=node.get("castKind"))
            return arg(0)
        if kind == "CStyleCastExpr":
            if node.get("castKind") == "NoOp" and dtype(node) in {"int", "float"}:
                return arg(0)
            require(node.get("castKind") == "BitCast", "unexpected attention explicit cast")
            return f"cast[{dtype(node)}]{arg(0)}"
        if kind == "DeclRefExpr":
            declaration = node["referencedDecl"]
            if declaration["kind"] in {"FunctionDecl", "CXXMethodDecl"}:
                name = declaration["name"]
                if name in {"cp_async_gs", "cp_async_wait"}:
                    # Clang resolves the specialization; the spelling below also
                    # checks the immediate operands of the async copy helpers.
                    template = node.get("foundReferencedDecl", {})
                    require(template.get("kind") == "FunctionTemplateDecl",
                            "expected a resolved async copy helper")
                    limit = node["range"]["begin"]["offset"]
                    end = node["range"]["end"]
                    raw = self.source[limit:end["offset"] + end["tokLen"]]
                    expect = b"tl::cp_async_gs<16>" if name == "cp_async_gs" else b"tl::cp_async_wait<0>"
                    require(b"".join(raw.split()) == expect, "unexpected async copy width or wait group")
                return name
            require(declaration["id"] in self.identities, "unknown reference in attention schedule")
            return self.identities[declaration["id"]]
        if kind == "PseudoObjectExpr":
            found = tuple(x for x in walk(node) if x["kind"] == "DeclRefExpr")
            members = {x.get("name") for x in walk(node) if x["kind"] == "MemberExpr"}
            require(found and all(x["referencedDecl"]["name"] == "threadIdx"
                    and dtype(x) == "const __cuda_builtin_threadIdx_t" for x in found)
                    and members == {"__fetch_builtin_x"}, "attention owner must use threadIdx.x")
            return "tid"
        if kind == "DeclStmt":
            require(len(parts) == 1 and parts[0]["kind"] == "VarDecl", "expected one private declaration")
            declaration = parts[0]
            require(not declaration.get("storageClass") and len(children(declaration)) == 1,
                    "expected a single local initializer")
            name = f"local{sum(x.startswith('local') for x in self.identities.values())}"
            self.identities[declaration["id"]] = name
            return f"{name}:{dtype(declaration)}={self.node(children(declaration)[0])}"
        if kind == "ForStmt":
            require(len(parts) == 5 and parts[1] == {} and parts[0]["kind"] == "DeclStmt"
                    and len(children(parts[0])) == 1, "expected a static ordered CUDA loop")
            induction = children(parts[0])[0]
            require(induction["kind"] == "VarDecl" and dtype(induction) == "int"
                    and not induction.get("storageClass") and len(children(induction)) == 1
                    and self.node(children(induction)[0]) == "0", "expected a zero-based attention loop")
            saved = self.identities.copy()
            name = f"i{self.loop_depth}"
            self.loop_depth += 1
            try:
                self.identities[induction["id"]] = name
                return f"for({arg(2)},{arg(3)}){arg(4)}"
            finally:
                self.loop_depth -= 1
                self.identities = saved
        if kind == "AttributedStmt":
            require(len(parts) == 2 and parts[0]["kind"] == "LoopHintAttr", "unexpected attention loop attribute")
            return arg(1)
        if kind == "CompoundStmt":
            saved = self.identities.copy()
            try:
                return "{" + ";".join(self.node(x) for x in parts) + "}"
            finally:
                self.identities = saved
        if kind == "IfStmt":
            require(len(parts) == 2, "unexpected else in attention schedule")
            return f"if({arg(0)}){arg(1)}"
        if kind == "BinaryOperator":
            return f"({arg(0)}{node['opcode']}{arg(1)})"
        if kind == "UnaryOperator":
            require(not node.get("isPostfix"), "unexpected postfix expression")
            return f"{node['opcode']}{arg(0)}"
        if kind == "ArraySubscriptExpr":
            return f"{arg(0)}[{arg(1)}]"
        if kind == "CXXMemberCallExpr":
            require(len(parts) == 1 and parts[0]["kind"] == "MemberExpr"
                    and parts[0].get("name") == "operator float" and dtype(node) == "float"
                    and len(children(parts[0])) == 1,
                    "expected CUTLASS half-to-f32 conversion")
            value = children(parts[0])[0]
            require(value["type"].get("desugaredQualType", dtype(value)) in
                    {"cutlass::half_t", "const cutlass::half_t"}, "expected binary16 conversion operand")
            return f"widen({self.node(value)})"
        if kind in {"CallExpr", "CXXOperatorCallExpr"}:
            return f"{arg(0)}(" + ",".join(self.node(x) for x in parts[1:]) + ")"
        if kind == "IntegerLiteral":
            return str(int(node["value"]))
        if kind == "FloatingLiteral":
            require(dtype(node) == "float", "expected f32 literal")
            return repr(float(node["value"]))
        if kind == "CXXConstructExpr":
            return f"ctor[{dtype(node)}](" + ",".join(self.node(x) for x in parts) + ")"
        raise LiftError("capability_missing", f"unhandled CUDA attention AST node {kind}")


def loop(level, bound, *body):
    return f"for((i{level}<{bound}),++i{level})" + "{" + ";".join(body) + "}"


def narrow(src, src_offset, dest, dest_offset):
    return (
        "local0:uint2=ctor[uint2]()",
        f"local1:float4=ctor[float4](*cast[float4 *]({src}+{src_offset}))",
        "operator=(cast[half2 *]&local0[0],__float22half2_rn(ctor[float2](cast[float2 *]&local1[0])))",
        "operator=(cast[half2 *]&local0[1],__float22half2_rn(ctor[float2](cast[float2 *]&local1[1])))",
        f"operator=(*cast[uint2 *]({dest}+{dest_offset}),local0)",
    )


def expected_schedule(rows, keys, depth, features, lanes, scale):
    """Storage staging, ordered score formation, normalization, PV and output."""
    sh = "cast[half_t *]shared"
    stride = lanes * 8
    vector = f"((i0*{stride})+(tid*8))"
    qsize, ksize, vsize = rows * depth, keys * depth, keys * features
    vb = qsize + ksize
    row = f"(tid&{rows - 1})"
    negative = "-__int_as_float(2139095040)"
    zero_f = "0.0"

    def fill(name, width):
        return loop(0, width // 4, f"local0:float={zero_f}",
                    f"operator=(*cast[float4 *]({name}+(i0*4)),make_float4(local0,local0,local0,local0))")

    pack_out = loop(1, 4, *narrow("output_acc", "((i0*16)+(i1*4))", "packed", "(i1*4)"))
    return [
        loop(0, qsize // stride, f"operator=(*cast[uint4 *]({sh}+{vector}),*cast[uint4 *](Q+{vector}))"),
        fill("output_acc", features),
        f"(denom[0]={zero_f})",
        f"(peak[0]={negative})",
        loop(0, ksize // stride, f"cp_async_gs(&{sh}[({vector}+{qsize})],&K[{vector}])"),
        "cp_async_commit()",
        loop(0, vsize // stride, f"cp_async_gs(&{sh}[({vector}+{vb})],&V[{vector}])"),
        "cp_async_commit()",
        fill("scores", keys),
        "cp_async_wait()",
        "__syncthreads()",
        loop(0, depth, loop(1, keys,
            f"(scores[i1]=(scores[i1]+(widen({sh}[(({row}*{depth})+i0)])*widen({sh}[(((i1*{depth})+i0)+{qsize})]))))")),
        "(previous[0]=peak[0])",
        f"(peak[0]={negative})",
        f"(maximum[0]={negative})",
        loop(0, keys, "(maximum[0]=max(maximum[0],scores[i0]))"),
        "(peak[0]=max(peak[0],maximum[0]))",
        "(peak[0]=max(peak[0],previous[0]))",
        f"(rescale[0]=exp2f(((previous[0]*{scale})-(peak[0]*{scale}))))",
        loop(0, keys, f"(scores[i0]=exp2f(((scores[i0]*{scale})-(peak[0]*{scale}))))"),
        f"(mass[0]={zero_f})",
        loop(0, keys, "(mass[0]=(mass[0]+scores[i0]))"),
        "(denom[0]=((denom[0]*rescale[0])+mass[0]))",
        loop(0, keys // 4, *narrow("scores", "(i0*4)", "probs", "(i0*4)")),
        loop(0, features, "(output_acc[i0]=(output_acc[i0]*rescale[0]))"),
        "cp_async_wait()",
        "__syncthreads()",
        loop(0, features, loop(1, keys,
            f"(output_acc[i0]=(output_acc[i0]+(widen(probs[i1])*widen({sh}[(((i1*{features})+i0)+{vb})]))))")),
        loop(0, features, "(output_acc[i0]=(output_acc[i0]/denom[0]))"),
        f"if(((tid>>{rows.bit_length() - 1})==0))" + "{" + loop(0, features // 16, pack_out,
            f"store_global_256(&*cast[ulonglong4 *](O+(({row}*{features})+(i0*16))),*cast[ulonglong4 *](packed+0))") + "}",
    ]


def capture(parsed, context, registry):
    function = parsed["function"]
    # Dispatch on the structural combination of vector stores and async staging.
    calls = {node.get("referencedDecl", {}).get("name") for node in walk(function) if node.get("kind") == "DeclRefExpr"}
    if not {"store_global_256", "cp_async_gs", "exp2f"} <= calls:
        return parsed
    contract = parsed["contract"]
    params = contract["parameters"]
    declarations = [p for p in children(function) if p["kind"] == "ParmVarDecl"]
    require(len(declarations) == len(params) == 4 and all(p["dtype"] == "f16" and len(p["shape"]) == 4
            and p["shape"][:2] == [1, 1] for p in params), "attention needs four static f16 rank-four views")
    require([d["name"] for d in declarations] == [p["name"] for p in params]
            and all(dtype(d).replace("__restrict", "").strip() ==
                    ("const half_t *" if p["mode"] == "input" else "half_t *")
                    for d, p in zip(declarations, params)), "attention ABI differs from its CUDA declaration")
    inputs = [i for i, p in enumerate(params) if p["mode"] == "input"]
    outputs = [i for i, p in enumerate(params) if p["mode"] == "output"]
    require(len(inputs) == 3 and len(outputs) == 1, "attention requires three inputs and one output")
    lanes = contract["block"][0]
    require(contract["grid"] == [1, 1, 1] and contract["block"][1:] == [1, 1] and 0 < lanes <= 1024,
            "attention lowering requires one CTA")
    block = next(node for node in children(function) if node["kind"] == "CompoundStmt")
    statements = children(block)
    labels = ["shared", "output_acc", "denom", "peak", "scores", "previous", "maximum",
              "rescale", "mass", "probs", "packed"]
    require(len(statements) == 41, "unrecognized attention staging or arithmetic schedule")
    locals_ = []
    for statement in statements[:len(labels)]:
        require(statement["kind"] == "DeclStmt" and len(children(statement)) == 1
                and children(statement)[0]["kind"] == "VarDecl", "unexpected attention declarations")
        locals_.append(children(statement)[0])
    shared = locals_[0]
    alignment = [c for c in walk(shared) if c["kind"] == "AlignedAttr"]
    require(shared.get("storageClass") == "extern"
            and {c["kind"] for c in children(shared)} == {"CUDASharedAttr", "AlignedAttr"}
            and len(alignment) == 1 and any(int(c["value"]) >= 16 for c in walk(alignment[0])
                                           if c["kind"] == "IntegerLiteral"), "invalid shared staging declaration")
    floats = [node for node in walk(statements[29]) if node["kind"] == "FloatingLiteral"]
    require(len(floats) == 2, "attention rescale requires explicit f32 coefficients")
    scale = float(floats[0]["value"])
    require(math.isfinite(scale) and scale > 0, "attention scale must be positive and finite")
    source = Path(contract["source"]).read_bytes()
    best = None

    for q, k, v in itertools.permutations(inputs):
        out = outputs[0]
        rows, depth = params[q]["shape"][2:]
        keys, key_depth = params[k]["shape"][2:]
        value_keys, features = params[v]["shape"][2:]
        if (depth != key_depth or keys != value_keys or params[out]["shape"][2:] != [rows, features]
                or min(rows, keys, depth, features) <= 0 or rows & (rows - 1) or lanes % rows
                or features % 16 or keys % 4):
            continue
        counts = (rows * depth, keys * depth, keys * features)
        if any(count % (lanes * 8) for count in counts) or contract["shared_mem"] < 2 * sum(counts):
            continue
        types = ["unsigned char[]", f"float[{features}]", "float[1]", "float[1]", f"float[{keys}]",
                 "float[1]", "float[1]", "float[1]", "float[1]", f"half_t[{keys}]", "half_t[16]"]
        if [dtype(d) for d in locals_] != types:
            continue
        require(all(not d.get("storageClass") and not children(d) for d in locals_[1:9])
                and all(not d.get("storageClass") and len(children(d)) == 1
                        and children(d)[0]["kind"] == "CXXConstructExpr" and not children(children(d)[0])
                        for d in locals_[9:]), "attention locals have unexpected initializers")
        identities = {d["id"]: label for d, label in zip(locals_, labels)}
        identities.update({declarations[i]["id"]: role for i, role in zip((q, k, v, out), ("Q", "K", "V", "O"))})
        normalizer = Schedule(identities, source)
        expected = expected_schedule(rows, keys, depth, features, lanes, repr(scale))
        for i, (stmt, target) in enumerate(zip(statements[len(labels):], expected), start=len(labels)):
            observed = normalizer.node(stmt)
            if observed != target:
                if best is None or i > best:
                    best = i
                break
        else:
            return {**parsed, "staged_attention": {
                "parameters": params, "shape": params[out]["shape"], "store": out,
                "expression": {"op": "staged_attention", "q": q, "k": k, "v": v, "scale": scale},
            }}
    raise LiftError("capability_missing", "CUDA attention staging or numeric schedule differs from the checked subset",
                    phase=best)


def register(registry):
    registry.stage("cuda.attention", "cuda_ast", "cuda_ast", capture)
