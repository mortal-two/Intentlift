"""Emit author-language source from the checked logical member representation."""
import ast
from intentlift.core.errors import require


def emit(program, context, registry):
    directions = {"input": "In", "output": "Out", "inout": "InOut"}
    annotations = [f"arg{i}: I.{directions[p['mode']]}[I.{p['dtype']}, {tuple(p['shape'])}]"
                   for i, p in enumerate(program["parameters"])]
    rank = len(program["shape"])
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
    expression = program["expression"]
    if expression["op"] == "ordered_product":
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
