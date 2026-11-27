"""Emit author-language source from the checked logical member representation."""
import ast
from intentlift.core.errors import require


def scan_statements(out, member, source, coordinates, axis, width):
    """Expand the inclusive warp stages without reassociating additions."""
    coordinate = coordinates[axis]

    def warp(prefix, base, wanted):
        lines, computed = [], {}

        def member_value(level, lane):
            if level == 0:
                address = list(coordinates)
                address[axis] = f"{base} + {lane}"
                return f"{source}[{', '.join(address)}]"
            offset = 1 << (level - 1)
            if lane < offset:
                return member_value(level - 1, lane)
            tag = level, lane
            if tag not in computed:
                a = member_value(level - 1, lane)
                b = member_value(level - 1, lane - offset)
                name = f"{prefix}_{level}_{lane}"
                lines.append(f"{name} = {a} + {b}")
                computed[tag] = name
            return computed[tag]

        values = [member_value(width.bit_length() - 1, lane) for lane in wanted]
        return lines, values

    before, (full,) = warp("complete", f"(segment * {width})", [width - 1])
    here, prefixes = warp("current", f"(({coordinate} // {width}) * {width})", range(width))
    value = prefixes[-1]
    for lane in reversed(range(width - 1)):
        value = f"({prefixes[lane]} if lane == {lane} else {value})"
    return [
        "carry = I.f32(0.0)",
        f"for segment in I.ordered({coordinate} // {width}):",
        *("    " + line for line in before),
        f"    carry = {full} + carry",
        f"lane = {coordinate} % {width}",
        *here,
        f"{out}[{member}] = {value} + carry",
    ]


def emit(program, context, registry):
    directions = {"input": "In", "output": "Out", "inout": "InOut"}
    annotations = [f"arg{i}: I.{directions[p['mode']]}[I.{p['dtype']}, {tuple(p['shape'])}]"
                   for i, p in enumerate(program["parameters"])]
    rank = len(program["shape"])
    expression = program["expression"]
    if expression["op"] == "staged_attention":
        require(rank == 4, "staged attention requires rank four")
        indices = ("batch", "head", "row", "col")
    elif expression["op"] == "staged_convolution":
        require(rank == 4, "staged convolution requires rank four")
        indices = ("batch", "row", "col", "feature")
    elif expression["op"] == "segmented_scan":
        require(1 <= rank <= 4, "scan emitter requires views of rank at most four")
        indices = tuple(f"i{axis}" for axis in range(rank))
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
        if op == "math":
            args = [expr(arg) for arg in node["args"]]
            name = node["name"]
            if name == "minimum_num":
                a, b = (f"(I.f32(-0.0) - {arg})" for arg in args)
                return f"(I.f32(-0.0) - I.maximum_num({a}, {b}))"
            return f"I.{name}({', '.join(args)})"
        require("spelling" in node and len(node.get("args", [])) == 2, "missing expression emitter")
        return f"({expr(node['args'][0])} {node['spelling']} {expr(node['args'][1])})"

    lines = ["import intentdsl as I", "", '@I.kernel(fast_math=False, denormal_fp_math="ieee")',
             "def lifted(" + ", ".join(annotations) + "):"]
    for axis, index in enumerate(indices):
        indent = "    " * (axis + 1)
        lines.append(f"{indent}for {index} in I.parallel({out}.dim({axis})):")
    indent = "    " * (rank + 1)
    if expression["op"] == "segmented_scan":
        source = f"arg{expression['source']}"
        scan = scan_statements(out, member, source, indices, expression["axis"], expression["segment"])
        lines.extend(indent + text for text in scan)
    elif expression["op"] == "staged_convolution":
        data, filt = (f"arg{expression[k]}" for k in ("data", "filter"))
        pad_h, pad_w = expression["padding"]
        load = f"I.cast({data}[batch, iy, ix, channel], I.f32)"
        for check in (f"ix < {data}.dim(2)", "ix >= 0", f"iy < {data}.dim(1)", "iy >= 0"):
            load = f"({load} if {check} else I.f32(0.0))"
        lines.extend(indent + body for body in [
            "acc = I.f32(0.0)",
            f"for term in I.ordered({expression['terms']}):",
            f"    filter_row = term // ({filt}.dim(1) * {data}.dim(3))",
            f"    filter_col = (term // {data}.dim(3)) % {filt}.dim(1)",
            f"    channel = term % {data}.dim(3)",
            f"    iy = row + filter_row - {pad_h}",
            f"    ix = col + filter_col - {pad_w}",
            f"    lhs = {load}",
            f"    rhs = I.cast({filt}[filter_row, filter_col, channel, feature], I.f32) "
            f"if filter_row < {filt}.dim(0) else I.f32(0.0)",
            "    acc = acc + (lhs * rhs)",
            f"{out}[{member}] = I.cast(acc, I.f16)",
        ])
    elif expression["op"] == "staged_attention":
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
        bound = factors[0]
        terms = []
        for factor in factors:
            address = "i, k" if factor["matrix"] else "k"
            value = f"arg{factor['slot']}[{address}]"
            terms.append(f"I.cast({value}, I.f32)" if factor.get("widen") else value)
        term = f"({' * '.join(terms)})" if len(terms) > 1 else terms[0]
        update = f"I.maximum_num(acc, {term})" if expression["combine"] == "maximum" else f"acc + {term}"
        lines.extend([
            f"{indent}acc = I.f32({expression['initial']!r})",
            f"{indent}for k in I.ordered(arg{bound['slot']}.dim({int(bound['matrix'])})):",
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
