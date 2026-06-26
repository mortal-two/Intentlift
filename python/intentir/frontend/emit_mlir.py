"""Emit `intent`-dialect MLIR from an IntentIR JSON function.

This is the ``JSON -> intent MLIR dialect`` front-end leg. It walks the
:class:`~intentir.ir.model.IntentFunction` body generically (it is driven by the
op list, not hard-coded to FlashAttention) and prints `intent`-dialect text at
concrete shapes, doing einsum/broadcast/reduce shape inference that matches the
numpy interpreter (``intentir.verify.interpret``).

Two artifacts are produced from the *same* JSON:

* :func:`emit_loop_module` — the whole function, with the structured ``intent.loop``
  carrying the online-softmax state. Proves the JSON is representable in the real
  dialect and lowers to standard MLIR (``-convert-intent-to-linalg`` -> ``scf.for``).
* :func:`emit_body_function` — the loop *body* as a standalone ``func.func`` whose
  args are the scan tiles + carries. This is the executable artifact: a host
  driver runs it once per K-tile (carries in persistent CUDA buffers), exactly
  the non-fused tiled FlashAttention that runs on the 5090D.

Instantiation: symbolic dims are bound to concrete ints; a configurable number of
leading **batch** dims (``fold`` — B,H for the flash JSON, asserted size 1) are
folded out so a single attention head is emitted as 2-D ops (the proven
executable form). Scalar inputs (``scale``) are emitted as rank-0 ``tensor<f32>``.
"""

from __future__ import annotations

from intentir.ir.model import IntentFunction, IntentOp

_DT = "f32"  # execute in f32 even where the JSON declares f16


def _resolve(shape, extents) -> list[int]:
    return [int(extents[s]) if isinstance(s, str) else int(s) for s in shape]


def _tt(shape: list[int]) -> str:
    """MLIR ranked-tensor type for a (possibly rank-0) shape."""
    if not shape:
        return f"tensor<{_DT}>"
    return "tensor<" + "x".join(str(s) for s in shape) + f"x{_DT}>"


def _strip(group: str, fold: int) -> str:
    """Drop the leading `fold` (batch) labels from one einsum label group."""
    return group[fold:]


def _broadcast(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    """numpy-style right-aligned broadcast of two shapes (None = unknown)."""
    if a is None:
        return b
    if b is None:
        return a
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + list(a)
    b = [1] * (n - len(b)) + list(b)
    return [max(x, y) for x, y in zip(a, b)]


def _infer_op(op: IntentOp, env: dict[str, list[int]], fold: int) -> list[int] | None:
    """Output shape of one body op given known operand shapes (None if not yet
    resolvable). Mirrors interpret.py's shape rules."""
    if op.op == "contraction":
        a, b = op.operands
        if a not in env or b not in env:
            return None
        ins, out = op.attrs["subscripts"].split("->")
        aL, bL = (s.strip() for s in ins.split(","))
        aL, bL, oL = _strip(aL, fold), _strip(bL, fold), _strip(out.strip(), fold)
        ext: dict[str, int] = {}
        for lab, shp in ((aL, env[a]), (bL, env[b])):
            for c, d in zip(lab, shp):
                ext[c] = d
        return [ext[c] for c in oL]
    if op.op == "reduce":
        (x,) = op.operands
        if x not in env:
            return None
        shp = list(env[x])
        for ax in (int(a) - fold for a in op.attrs["axes"]):  # keepdims
            shp[ax] = 1
        return shp
    if op.op == "const":
        return list(op.attrs.get("shape", []))
    if op.op == "cast":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_unary":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_binary":
        a, b = op.operands
        return _broadcast(env.get(a), env.get(b))
    if op.op == "select":
        c, x, y = op.operands
        return _broadcast(_broadcast(env.get(c), env.get(x)), env.get(y))
    raise NotImplementedError(f"emit_mlir: unhandled body op {op.op!r}")


_CMP_FNS = {"gt", "ge", "lt", "le", "eq"}


def _infer_body(body: list[IntentOp], env: dict[str, list[int]], fold: int) -> None:
    for _ in range(len(body) + 2):
        changed = False
        for op in body:
            sh = _infer_op(op, env, fold)
            if sh is not None and env.get(op.result) != sh:
                env[op.result] = sh
                changed = True
        if not changed:
            break


class _Printer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._n = 0
        self.ssa: dict[str, str] = {}   # JSON name -> SSA name
        self.bool_results: set[str] = set()  # JSON names that are i1 (cmp outputs)

    def _ttn(self, name: str, env: dict[str, list[int]]) -> str:
        """Tensor type for a JSON value, i1 if it is a comparison result."""
        if name not in self.bool_results:
            return _tt(env[name])
        shape = env[name]
        if not shape:
            return "tensor<i1>"
        return "tensor<" + "x".join(str(s) for s in shape) + "xi1>"

    def fresh(self, name: str) -> str:
        self._n += 1
        s = f"%{name}_{self._n}"
        self.ssa[name] = s
        return s

    def arg(self, name: str) -> str:
        """Clean SSA name for a function/block arg (no numeric suffix), so the
        orchestrator can key inputs by the original JSON name."""
        s = f"%{name}"
        self.ssa[name] = s
        return s

    def emit_op(self, op: IntentOp, env: dict[str, list[int]], fold: int,
                indent: str = "    ") -> None:
        a = op.attrs
        if op.op == "cast":
            # The corpus runs entirely in f32, so a dtype cast is a value-level
            # no-op: alias the operand SSA. (A precision-changing cast would need
            # arith/tensor cast ops; out of scope for the f32 lifting check.)
            self.ssa[op.result] = self.ssa[op.operands[0]]
            return
        r = self.fresh(op.result)
        if op.op == "contraction":
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = _tt(env[op.operands[0]]), _tt(env[op.operands[1]])
            ins, out = a["subscripts"].split("->")
            aL, bL = (s.strip() for s in ins.split(","))
            sub = f"{_strip(aL, fold)},{_strip(bL, fold)}->{_strip(out.strip(), fold)}"
            self.lines.append(
                f'{indent}{r} = intent.contraction {x}, {y} '
                f'{{subscripts = "{sub}"}} : ({tx}, {ty}) -> {_tt(env[op.result])}')
        elif op.op == "reduce":
            x = self.ssa[op.operands[0]]
            axes = ", ".join(str(int(ax) - fold) for ax in a["axes"])
            self.lines.append(
                f'{indent}{r} = intent.reduce {x} {{fn = "{a["fn"]}", '
                f'axes = [{axes}], assoc = "{a["assoc"]}"}} : '
                f'({_tt(env[op.operands[0]])}) -> {_tt(env[op.result])}')
        elif op.op == "map_unary":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.map_unary {x} {{fn = "{a["fn"]}"}} : '
                f'({_tt(env[op.operands[0]])}) -> {_tt(env[op.result])}')
        elif op.op == "map_binary":
            if a["fn"] in _CMP_FNS:
                self.bool_results.add(op.result)
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.map_binary {x}, {y} {{fn = "{a["fn"]}"}} : '
                f'({tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "select":
            c, x, y = (self.ssa[o] for o in op.operands)
            tc, tx, ty = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.select {c}, {x}, {y} : '
                f'({tc}, {tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "const":
            # A literal value -> standard arith.constant (no intent semantics to
            # lower). Splat dense over the (possibly rank-0) result tensor type.
            self.lines.append(
                f'{indent}{r} = arith.constant dense<{float(a["value"])!r}> : '
                f'{_tt(env[op.result])}')
        else:
            raise NotImplementedError(f"emit_mlir: unhandled op {op.op!r}")


def _tile(loop: IntentOp, extents: dict[str, int]) -> int:
    th = loop.trip.get("tile_hint") if loop.trip else None
    if isinstance(th, str):
        return int(extents[th])
    return int(th) if th is not None else 1


def _setup(fn: IntentFunction, dims: dict[str, int], fold: int):
    """Shape inference shared by both emitters.

    Returns (loop, env, carries, yields, scan_specs, tile) where scan_specs is a
    list of (slice_name, source_input_name, folded_axis).
    """
    loop = next(o for o in fn.body if o.op == "loop")
    extents = dict(dims)
    env: dict[str, list[int]] = {}
    for t in fn.inputs:
        full = _resolve(t.shape, extents)
        # Folding leading batch dims is only valid when they are size 1 (a single
        # attention head). Refuse to silently miscompile a B>1 / H>1 instance.
        if len(full) >= fold and any(d != 1 for d in full[:fold]):
            raise ValueError(
                f"emit_mlir: cannot fold {fold} leading dims of {t.name} "
                f"(shape {full}); folded dims must be size 1")
        env[t.name] = full[fold:]
    tile = _tile(loop, extents)
    scan_specs = []
    for s in loop.scan:
        ax = int(s["axis"]) - fold
        slshape = list(env[s["from"]])
        slshape[ax] = tile
        env[s["name"]] = slshape
        scan_specs.append((s["name"], s["from"], ax))
    carries = [c["name"] for c in loop.carry]
    yields = list(loop.yields)
    for _ in range(len(loop.body) + 3):
        _infer_body(loop.body, env, fold)
        moved = False
        for cname, yname in zip(carries, yields):
            if yname in env and env.get(cname) != env[yname]:
                env[cname] = list(env[yname])
                moved = True
        if not moved:
            break
    return loop, env, carries, yields, scan_specs, tile


def emit_body_function(fn: IntentFunction, dims: dict[str, int], fold: int = 2,
                       func_name: str = "flash_body_fwd") -> str:
    """The loop body as an executable ``func.func`` (scan tiles + carries in,
    next carries out). This is what the host-driven trip loop launches."""
    loop, env, carries, yields, scan_specs, _ = _setup(fn, dims, fold)
    p = _Printer()

    referenced: set[str] = set()
    for op in loop.body:
        referenced.update(op.operands)
    # Outer inputs the body reads (Q, scale, ...): function inputs that appear in
    # the body directly (the scanned tensors enter via their slices, below).
    outer = [t.name for t in fn.inputs if t.name in referenced]

    args: list[str] = []
    for nm in outer:
        args.append(f"{p.arg(nm)}: {_tt(env[nm])}")
    for sname, _src, _ax in scan_specs:
        args.append(f"{p.arg(sname)}: {_tt(env[sname])}")
    for c in carries:
        args.append(f"{p.arg(c)}: {_tt(env[c])}")
    rty = ", ".join(_tt(env[c]) for c in carries)

    for op in loop.body:
        p.emit_op(op, env, fold)
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(_tt(env[y]) for y in yields)

    head = (f"// Generated by intentir.frontend.emit_mlir from "
            f"{fn.name} (single-head, fold={fold}).\n"
            f"// One K-tile of the FlashAttention online-softmax recurrence.\n"
            f"module {{\n"
            f"  func.func @{func_name}({', '.join(args)}) -> ({rty}) {{")
    tail = (f"    return {yvals} : {ytys}\n  }}\n}}\n")
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


def emit_loop_module(fn: IntentFunction, dims: dict[str, int], fold: int = 2,
                     func_name: str = "flash_fwd") -> str:
    """The whole function with the structured ``intent.loop``. Representability +
    lowers to ``scf.for`` via ``-convert-intent-to-linalg``."""
    loop, env, carries, yields, scan_specs, tile = _setup(fn, dims, fold)
    p = _Printer()

    # Function signature: the IntentIR function inputs (folded, scalars rank-0).
    args = [f"{p.arg(t.name)}: {_tt(env[t.name])}" for t in fn.inputs]
    # Carry inits as function args too (so the module is self-contained and the
    # init *values* — NEG_INF/0/zeros — stay the caller's responsibility).
    init_names = [f"{c}_init" for c in carries]
    for c, init in zip(carries, init_names):
        env[init] = env[c]
        args.append(f"{p.arg(init)}: {_tt(env[c])}")

    scan_ssa = ", ".join(p.ssa[s["from"]] for s in loop.scan)
    scan_tys = ", ".join(_tt(env[s["from"]]) for s in loop.scan)
    init_ssa = ", ".join(p.ssa[i] for i in init_names)
    init_tys = ", ".join(_tt(env[c]) for c in carries)
    scan_axes = ", ".join(f"{int(s['axis']) - fold} : i64" for s in loop.scan)
    res_tys = ", ".join(_tt(env[c]) for c in carries)

    # Body region block args: scan slices then carries.
    bargs = []
    for sname, _src, _ax in scan_specs:
        bargs.append(f"{p.arg(sname)}: {_tt(env[sname])}")
    for c in carries:
        # carry block arg shadows the init binding for the body
        bargs.append(f"{p.arg(c)}: {_tt(env[c])}")
    body_lines: list[str] = []
    saved, p.lines = p.lines, body_lines
    for op in loop.body:
        p.emit_op(op, env, fold, indent="      ")
    p.lines = saved
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(_tt(env[y]) for y in yields)

    out_res = next((o for o in fn.body if o.op != "loop"), None)
    loop_res = p.fresh("loop")

    lines = [
        f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
        f"(single-head, fold={fold}).",
        "module {",
        f"  func.func @{func_name}({', '.join(args)}) -> ({res_tys}) {{",
        f"    {loop_res}:{len(carries)} = intent.loop "
        f"scan({scan_ssa} : {scan_tys}) init({init_ssa} : {init_tys}) "
        f"{{scan_axes = [{scan_axes}], tile_hint = {tile} : i64}} "
        f"-> {res_tys} do {{",
        f"    ^bb0({', '.join(bargs)}):",
        *body_lines,
        f"      intent.yield {yvals} : {ytys}",
        "    }",
    ]
    rets = [f"{loop_res}#{i}" for i in range(len(carries))]
    lines.append(f"    return {', '.join(rets)} : {res_tys}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_module(fn: IntentFunction, dims: dict[str, int], fold: int = 0,
                func_name: str | None = None) -> str:
    """A flat (loop-free) IntentFunction as an executable ``func.func``.

    Inputs become function args (clean SSA names so the orchestrator keys them by
    JSON name), the atomic op-graph is emitted in order, and the declared outputs
    are returned. Covers softmax / layernorm / rmsnorm / silu / gelu / matmul /
    elementwise — every kernel whose body is a straight-line tensor graph.
    """
    if any(o.op == "loop" for o in fn.body):
        raise ValueError("emit_module: function has a loop; use emit_loop_module")
    extents = dict(dims)
    env: dict[str, list[int]] = {}
    for t in fn.inputs:
        env[t.name] = _resolve(t.shape, extents)[fold:]
    _infer_body(fn.body, env, fold)

    p = _Printer()
    args = [f"{p.arg(t.name)}: {_tt(env[t.name])}" for t in fn.inputs]
    for op in fn.body:
        p.emit_op(op, env, fold)
        if op.out is not None:                 # destination-passing alias
            p.ssa[op.out] = p.ssa[op.result]
            env[op.out] = env[op.result]

    outs = []
    for t in fn.outputs:
        if t.name not in p.ssa:
            raise ValueError(f"emit_module: output {t.name!r} never produced")
        outs.append(t.name)
    ovals = ", ".join(p.ssa[n] for n in outs)
    otys = ", ".join(_tt(env[n]) for n in outs)
    name = func_name or fn.name

    head = (f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
            f"(flat graph, fold={fold}).\n"
            f"module {{\n"
            f"  func.func @{name}({', '.join(args)}) -> ({otys}) {{")
    tail = f"    return {ovals} : {otys}\n  }}\n}}\n"
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


__all__ = ["emit_body_function", "emit_loop_module", "emit_module"]
