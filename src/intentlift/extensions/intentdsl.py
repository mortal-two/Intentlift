"""Emit author-language source from the checked logical member representation."""
import ast
from intentlift.core.errors import require


def emit(program, context, registry):
    directions = {"input": "In", "output": "Out", "inout": "InOut"}
    annotations = [f"arg{i}: I.{directions[p['mode']]}[I.{p['dtype']}, {tuple(p['shape'])}]"
                   for i, p in enumerate(program["parameters"])]
    rank = len(program["shape"])
    expression = program["expression"]
    if expression["op"] == "staged_attention":
        require(rank == 4, "staged attention requires rank four")
        indices = ("batch", "head", "row", "col")
    else:
        require(rank in {1, 2}, "pointwise emitter requires rank one or two")
        indices = ("i",) if rank == 1 else ("row", "col")
    member = ", ".join(indices)
    out = f"arg{program['store']}"

    def expr(node):
        op = node["op"]
        if op == "read":
            coordinates = ", ".join(indices[axis] for axis in node["axes"])
            return f"arg{node['slot']}[{coordinates}]"
        if op == "literal":
            return f"I.f32({node['value']!r})"
        if op == "cast":
            return f"I.cast({expr(node['args'][0])}, I.{node['dtype']})"
        if op == "neg":
            return f"(I.f32(-0.0) - {expr(node['args'][0])})"
        if op == "select":
            cond, yes, no = (expr(arg) for arg in node["args"])
            return f"({yes} if {cond} else {no})"
        require("spelling" in node and len(node.get("args", [])) == 2, "missing expression emitter")
        return f"({expr(node['args'][0])} {node['spelling']} {expr(node['args'][1])})"

    lines = ["import intentdsl as I", "", '@I.kernel(fast_math=False, denormal_fp_math="ieee")',
             "def lifted(" + ", ".join(annotations) + "):"]
    for axis, index in enumerate(indices):
        indent = "    " * (axis + 1)
        lines.append(f"{indent}for {index} in I.parallel({out}.dim({axis})):")
    indent = "    " * (rank + 1)
    if expression["op"] == "staged_attention":
        q, k, v = (f"arg{expression[role]}" for role in ("q", "k", "v"))
        score = [
            "    score = I.f32(0.0)",
            f"    for d in I.ordered({q}.dim(3)):",
            f"        score = score + (I.cast({q}[batch, head, row, d], I.f32)"
            f" * I.cast({k}[batch, head, token, d], I.f32))",
        ]
        lines.extend(indent + body for body in [
            f"scale = I.f32({expression['scale']!r})",
            "previous = -I.inf",
            "peak = -I.inf",
            f"for token in I.ordered({k}.dim(2)):",
            *score,
            "    peak = I.maximum_num(peak, score)",
            "peak = I.maximum_num(I.maximum_num(-I.inf, peak), previous)",
            "rescale = I.exp2((previous * scale) - (peak * scale))",
            "weighted = I.f32(0.0) * rescale",
            "mass = I.f32(0.0)",
            f"for token in I.ordered({k}.dim(2)):",
            *score,
            "    weight = I.exp2((score * scale) - (peak * scale))",
            "    mass = mass + weight",
            "    rounded = I.cast(I.cast(weight, I.f16), I.f32)",
            f"    weighted = weighted + (rounded * I.cast({v}[batch, head, token, col], I.f32))",
            "denom = (I.f32(0.0) * rescale) + mass",
            f"{out}[{member}] = I.cast(weighted / denom, I.f16)",
        ])
    elif expression["op"] == "indexed_selection":
        src = f"arg{expression['input']}"
        idx_out = f"arg{expression['indices']}"
        masked = f"(I.f32({expression['sentinel']!r}) if candidate >= threshold else candidate)"
        lines.extend(indent + body for body in [
            "threshold = I.inf",
            "selected = -I.inf",
            f"index = I.i32({expression['empty_index']})",
            "for step in I.ordered(col + 1):",
            "    selected = -I.inf",
            f"    for j in I.ordered({src}.dim(1)):",
            f"        candidate = {src}[row, j]",
            f"        candidate = candidate if step == 0 else {masked}",
            "        selected = I.maximum_num(selected, candidate)",
            f"    index = I.i32({expression['empty_index']})",
            f"    for j in I.ordered({src}.dim(1)):",
            f"        candidate = {src}[row, j]",
            f"        candidate = candidate if step == 0 else {masked}",
            "        index = I.cast(j, I.i32) if candidate == selected else index",
            "    threshold = selected if selected < threshold else threshold",
            f"{out}[{member}] = selected",
            f"{idx_out}[{member}] = index",
        ])
    elif expression["op"] == "ordered_row":
        factors = expression["factors"]
        matrix = next(f["slot"] for f in factors if f["matrix"])
        terms = [f"arg{f['slot']}[{'i, k' if f['matrix'] else 'k'}]" for f in factors]
        if len(terms) == 2:
            term = f"(I.cast({terms[0]}, I.f32) * I.cast({terms[1]}, I.f32))"
        else:
            term = terms[0]
        update = f"I.maximum_num(acc, {term})" if expression["combine"] == "maximum" else f"acc + {term}"
        lines.extend([
            f"{indent}acc = I.f32({expression['initial']!r})",
            f"{indent}for k in I.ordered(arg{matrix}.dim(1)):",
            f"{indent}    acc = {update}",
            f"{indent}{out}[{member}] = acc",
        ])
    elif expression["op"] == "ordered_product":
        factors = expression["factors"]
        row_slot = next(f["slot"] for f in factors if f["axis"] == "row")
        coords = {"row": "row, k", "col": "k, col"}
        terms = [f"I.cast(arg{f['slot']}[{coords[f['axis']]}], I.f32)" for f in factors]
        lines.extend([
            f"{indent}acc = I.f32({expression['initial']!r})",
            f"{indent}for k in I.ordered(arg{row_slot}.dim(1)):",
            f"{indent}    acc = acc + ({terms[0]} * {terms[1]})",
            f"{indent}{out}[{member}] = acc",
        ])
    else:
        lines.append(f"{indent}{out}[{member}] = {expr(expression)}")
    return "\n".join(lines) + "\n"


def verify(source):
    require(isinstance(source, str), "IntentDSL source must be text", "extension_invalid")
    ast.parse(source)


def register(registry):
    registry.representation("intentdsl_source", verify)
    registry.stage("intentdsl.emit", "pointwise", "intentdsl_source", emit)
