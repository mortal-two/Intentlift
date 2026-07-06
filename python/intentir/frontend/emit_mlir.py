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


def _infer_op(op: IntentOp, env: dict[str, list[int]], fold: int,
              extents: dict[str, int] | None = None) -> list[int] | None:
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
    if op.op == "scan":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "arg_reduce":
        (x,) = op.operands
        if x not in env:
            return None
        ax = int(op.attrs["axis"]) - fold  # collapse the axis (rank-1 index out)
        return [d for i, d in enumerate(env[x]) if i != ax]
    if op.op == "sort":
        (x,) = op.operands
        if x not in env:
            return None
        shp = list(env[x])            # shape-preserving (values & indices)
        w = op.attrs.get("k")         # optional topk width along the last axis
        if w is not None:
            shp[-1] = int(w)
        return shp
    if op.op == "iota":
        return [int(op.attrs["n"])]
    if op.op == "const":
        return _resolve(op.attrs.get("shape", []), extents or {})
    if op.op == "rand":
        # Counter-based RNG (C5): shape carried in attrs (like const), symbolic
        # dims resolved against the extents, fold the leading batch dims out so a
        # single-head instance emits rank-2.
        return _resolve(op.attrs.get("shape", []), extents or {})[fold:]
    if op.op == "cast":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_unary":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_binary":
        a, b = op.operands
        return _broadcast(env.get(a), env.get(b))
    if op.op == "gather":
        d, idx = op.operands
        if d not in env or idx not in env:
            return None
        axis = int(op.attrs["axis"]) - fold
        if op.attrs.get("mode", "along_axis") == "along_axis":
            return list(env[idx])                        # shaped like index
        return list(env[d][:axis]) + list(env[idx]) + list(env[d][axis + 1:])
    if op.op == "scatter":
        d = op.operands[0]
        return list(env[d]) if d in env else None        # shaped like data
    if op.op == "nonzero":
        # Dynamic-shape output (C8): [nnz, ndim]. nnz is data-dependent, so the
        # emitter uses the MAX (upper-bound) extent = prod(cond shape); ndim =
        # cond rank. The true nnz is resolved host-side (the interpreter yields
        # the exact [nnz, ndim]); the emitted static type is an upper bound.
        (c,) = op.operands
        if c not in env:
            return None
        import math as _m
        return [_m.prod(env[c]) if env[c] else 1, len(env[c])]
    if op.op == "masked_select":
        d, m = op.operands
        if d not in env:
            return None
        import math as _m
        return [_m.prod(env[d]) if env[d] else 1]         # [nnz] upper bound
    if op.op == "masked_scatter":
        d = op.operands[0]
        return list(env[d]) if d in env else None         # data-shaped
    if op.op == "select":
        c, x, y = op.operands
        return _broadcast(_broadcast(env.get(c), env.get(x)), env.get(y))
    if op.op == "concat":
        if any(o not in env for o in op.operands):
            return None
        axis = int(op.attrs["axis"]) - fold
        shp = list(env[op.operands[0]])
        shp[axis] = sum(env[o][axis] for o in op.operands)
        return shp
    if op.op == "pad":
        (x,) = op.operands
        if x not in env:
            return None
        lo = [int(v) for v in op.attrs["lo"]][fold:]
        hi = [int(v) for v in op.attrs["hi"]][fold:]
        return [d + lo[i] + hi[i] for i, d in enumerate(env[x])]
    if op.op == "tile":
        (x,) = op.operands
        if x not in env:
            return None
        reps = [int(v) for v in op.attrs["reps"]][fold:]
        return [d * reps[i] for i, d in enumerate(env[x])]
    if op.op in ("flip", "triu"):
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "pool":
        (x,) = op.operands
        if x not in env:
            return None
        N, C, H, W = env[x]
        kH, kW = int(op.attrs["kH"]), int(op.attrs["kW"])
        sH, sW = int(op.attrs["sH"]), int(op.attrs["sW"])
        padH, padW = int(op.attrs["padH"]), int(op.attrs["padW"])
        OH = (H + 2 * padH - kH) // sH + 1
        OW = (W + 2 * padW - kW) // sW + 1
        return [N, C, OH, OW]
    if op.op == "upsample":
        (x,) = op.operands
        if x not in env:
            return None
        N, C, H, W = env[x]
        sh, sw = (int(v) for v in op.attrs["scale"])
        return [N, C, H * sh, W * sw]
    if op.op == "polar":
        a0 = op.operands[0]
        return list(env[a0]) if a0 in env else None
    if op.op == "conv":
        xn, wn = op.operands[0], op.operands[1]
        if xn not in env or wn not in env:
            return None
        inshape, wshape = env[xn], env[wn]
        R = len(inshape) - 2
        stride = [int(v) for v in op.attrs["stride"]]
        pad = [int(v) for v in op.attrs["padding"]]
        dil = [int(v) for v in op.attrs["dilation"]]
        out = [inshape[0], wshape[0]]
        for j in range(R):
            H, k = inshape[2 + j], wshape[2 + j]
            out.append((H + 2 * pad[j] - dil[j] * (k - 1) - 1) // stride[j] + 1)
        return out
    raise NotImplementedError(f"emit_mlir: unhandled body op {op.op!r}")


_CMP_FNS = {"gt", "ge", "lt", "le", "eq", "ne",
            "logical_and", "logical_or", "logical_xor", "isclose"}
# map_unary fns whose result is an i1 (finiteness predicates + i1 complement).
_U_TO_I1 = {"isnan", "isinf", "isfinite", "logical_not"}


def _mlir_f(v) -> str:
    """Format a Python float as an MLIR-parseable float literal. MLIR requires a
    decimal point before the optional exponent, so bare scientific literals like
    ``1e-05`` (which ``repr`` emits) must become ``1.0e-05``."""
    s = repr(float(v))
    if "e" in s or "E" in s:
        e = "e" if "e" in s else "E"
        mant, _, exp = s.partition(e)
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{exp}"
    if "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def _infer_body(body: list[IntentOp], env: dict[str, list[int]], fold: int,
                extents: dict[str, int] | None = None) -> None:
    for _ in range(len(body) + 2):
        changed = False
        for op in body:
            sh = _infer_op(op, env, fold, extents)
            if sh is None:
                continue
            # Multi-result ops (sort -> values, indices) name every result; all
            # results share the inferred shape.
            names = op.result if isinstance(op.result, list) else [op.result]
            for nm in names:
                if env.get(nm) != sh:
                    env[nm] = sh
                    changed = True
        if not changed:
            break


class _Printer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._n = 0
        self.ssa: dict[str, str] = {}   # JSON name -> SSA name
        self.bool_results: set[str] = set()  # JSON names that are i1 (cmp outputs)
        self.int_results: dict[str, str] = {}  # JSON names that are int (iota)

    def _ttn(self, name: str, env: dict[str, list[int]]) -> str:
        """Tensor type for a JSON value: i1 if a comparison result, an integer
        width if an iota result, else the default f32."""
        elt = "i1" if name in self.bool_results else self.int_results.get(name)
        if elt is None:
            return _tt(env[name])
        shape = env[name]
        if not shape:
            return f"tensor<{elt}>"
        return "tensor<" + "x".join(str(s) for s in shape) + f"x{elt}>"

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
            to = a.get("to_dtype", "")
            if to in ("i1", "bool"):
                # A boolean cast is a truthiness test (x != 0), NOT a value-level
                # no-op — emit a real intent.cast so logical_not(x.bool()) & co.
                # carry a proper i1 through lowering.
                self.bool_results.add(op.result)
                rr = self.fresh(op.result)
                x = self.ssa[op.operands[0]]
                self.lines.append(
                    f'{indent}{rr} = intent.cast {x} {{to_dtype = "i1"}} : '
                    f'({self._ttn(op.operands[0], env)}) -> '
                    f'{self._ttn(op.result, env)}')
                return
            # The corpus otherwise runs entirely in f32, so a dtype cast is a
            # value-level no-op: alias the operand SSA. (A precision-changing
            # float cast would need arith/tensor cast ops; out of scope here.)
            self.ssa[op.result] = self.ssa[op.operands[0]]
            return
        if op.op == "sort":
            # The op ALWAYS has two results (sorted values + argsort indices); a
            # JSON op may name both (list result) or just one (values-only sort,
            # or indices-only argsort via attrs {"return":"indices"}). Emit both
            # SSA results, aliasing a throwaway name for the unnamed one.
            if isinstance(op.result, list):
                vname, iname = op.result[0], op.result[1]
            elif a.get("return") == "indices":
                vname, iname = None, op.result
            else:
                vname, iname = op.result, None
            named = iname if vname is None else vname
            shp = env[named]
            self._n += 1
            vssa = self.fresh(vname) if vname else f"%_sortv_{self._n}"
            issa = self.fresh(iname) if iname else f"%_sorti_{self._n}"
            if iname:
                self.int_results[iname] = "i32"
            x = self.ssa[op.operands[0]]
            desc = "true" if a.get("descending", False) else "false"
            vty = _tt(shp)
            ity = "tensor<" + "x".join(str(s) for s in shp) + "xi32>"
            axis = int(a.get("axis", len(env[op.operands[0]]) - 1 + fold)) - fold
            self.lines.append(
                f'{indent}{vssa}, {issa} = intent.sort {x} '
                f'{{descending = {desc}, axis = {axis} : i64}} : '
                f'({_tt(env[op.operands[0]])}) -> ({vty}, {ity})')
            return
        if op.op == "pool":
            # avg -> one f32 output; max may carry a 2nd i64 argmax-index result
            # (max_pool2d_with_indices). op.result is a list [out, indices] when
            # indices are named, else a single str.
            x = self.ssa[op.operands[0]]
            if isinstance(op.result, list):
                oname, iname = op.result[0], op.result[1]
            else:
                oname, iname = op.result, None
            oshp = env[oname]
            ossa = self.fresh(oname)
            oty = _tt(oshp)
            attrs = (f'fn = "{a["fn"]}", kH = {int(a["kH"])} : i64, '
                     f'kW = {int(a["kW"])} : i64, sH = {int(a["sH"])} : i64, '
                     f'sW = {int(a["sW"])} : i64, padH = {int(a["padH"])} : i64, '
                     f'padW = {int(a["padW"])} : i64')
            if iname is not None:
                self.int_results[iname] = "i64"
                issa = self.fresh(iname)
                ity = "tensor<" + "x".join(str(s) for s in oshp) + "xi64>"
                self.lines.append(
                    f'{indent}{ossa}, {issa} = intent.pool {x} {{{attrs}}} : '
                    f'({_tt(env[op.operands[0]])}) -> ({oty}, {ity})')
            else:
                self.lines.append(
                    f'{indent}{ossa} = intent.pool {x} {{{attrs}}} : '
                    f'({_tt(env[op.operands[0]])}) -> {oty}')
            return
        if op.op == "polar":
            # abs,angle -> (real, imag) real/imag PAIR (two f32 results).
            absv, ang = (self.ssa[o] for o in op.operands)
            rname, iname = op.result[0], op.result[1]
            rssa, issa = self.fresh(rname), self.fresh(iname)
            ta = _tt(env[op.operands[0]])
            self.lines.append(
                f'{indent}{rssa}, {issa} = intent.polar {absv}, {ang} : '
                f'({ta}, {ta}) -> ({_tt(env[rname])}, {_tt(env[iname])})')
            return
        r = self.fresh(op.result)
        if op.op == "conv":
            # input, weight, optional bias -> single output. stride/padding/
            # dilation are length-R spatial arrays; groups partitions channels.
            x = self.ssa[op.operands[0]]
            w = self.ssa[op.operands[1]]
            tx, tw = _tt(env[op.operands[0]]), _tt(env[op.operands[1]])
            st = ", ".join(str(int(v)) for v in a["stride"])
            pd = ", ".join(str(int(v)) for v in a["padding"])
            dl = ", ".join(str(int(v)) for v in a["dilation"])
            grp = int(a.get("groups", 1))
            attrs = (f'stride = [{st}], padding = [{pd}], dilation = [{dl}], '
                     f'groups = {grp} : i64')
            if len(op.operands) > 2:
                bn = op.operands[2]
                bssa, tb = self.ssa[bn], _tt(env[bn])
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w}, {bssa} {{{attrs}}} : '
                    f'({tx}, {tw}, {tb}) -> {_tt(env[op.result])}')
            else:
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w} {{{attrs}}} : '
                    f'({tx}, {tw}) -> {_tt(env[op.result])}')
        elif op.op == "upsample":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.upsample {x} {{mode = "nearest"}} : '
                f'({_tt(env[op.operands[0]])}) -> {_tt(env[op.result])}')
        elif op.op == "contraction":
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = _tt(env[op.operands[0]]), _tt(env[op.operands[1]])
            ins, out = a["subscripts"].split("->")
            aL, bL = (s.strip() for s in ins.split(","))
            sub = f"{_strip(aL, fold)},{_strip(bL, fold)}->{_strip(out.strip(), fold)}"
            self.lines.append(
                f'{indent}{r} = intent.contraction {x}, {y} '
                f'{{subscripts = "{sub}"}} : ({tx}, {ty}) -> {_tt(env[op.result])}')
        elif op.op == "reduce":
            # all/any are boolean reductions over an i1 input to an i1 result;
            # _ttn carries the i1 element type on both sides.
            if a["fn"] in ("all", "any"):
                self.bool_results.add(op.result)
            x = self.ssa[op.operands[0]]
            axes = ", ".join(str(int(ax) - fold) for ax in a["axes"])
            self.lines.append(
                f'{indent}{r} = intent.reduce {x} {{fn = "{a["fn"]}", '
                f'axes = [{axes}], assoc = "{a["assoc"]}"}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "scan":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.scan {x} {{fn = "{a["fn"]}", '
                f'axis = {int(a["axis"]) - fold}}} : '
                f'({_tt(env[op.operands[0]])}) -> {_tt(env[op.result])}')
        elif op.op == "arg_reduce":
            # Index-of-extremum: an INTEGER (i32) result with `axis` collapsed.
            x = self.ssa[op.operands[0]]
            self.int_results[op.result] = "i32"
            self.lines.append(
                f'{indent}{r} = intent.arg_reduce {x} {{fn = "{a["fn"]}", '
                f'axis = {int(a["axis"]) - fold} : i64}} : '
                f'({_tt(env[op.operands[0]])}) -> {self._ttn(op.result, env)}')
        elif op.op == "map_unary":
            # Predicate fns (isnan/isinf/isfinite/logical_not) yield an i1; every
            # other fn is element-type-preserving. Use _ttn so an i1 operand
            # (e.g. logical_not of an isnan result) prints its true type.
            if a["fn"] in _U_TO_I1:
                self.bool_results.add(op.result)
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.map_unary {x} {{fn = "{a["fn"]}"}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "map_binary":
            if a["fn"] in _CMP_FNS:
                self.bool_results.add(op.result)
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.map_binary {x}, {y} {{fn = "{a["fn"]}"}} : '
                f'({tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "select":
            # The result shares the x/y branch element type (verifier-enforced),
            # so propagate i1/int-ness of the branches onto the result.
            xn = op.operands[1]
            if xn in self.bool_results:
                self.bool_results.add(op.result)
            elif xn in self.int_results:
                self.int_results[op.result] = self.int_results[xn]
            c, x, y = (self.ssa[o] for o in op.operands)
            tc, tx, ty = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.select {c}, {x}, {y} : '
                f'({tc}, {tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "gather":
            d, i = (self.ssa[o] for o in op.operands)
            td, ti = (self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            mode = a.get("mode", "along_axis")
            self.lines.append(
                f'{indent}{r} = intent.gather {d}, {i} '
                f'{{axis = {axis} : i64, mode = "{mode}"}} : '
                f'({td}, {ti}) -> {self._ttn(op.result, env)}')
        elif op.op == "scatter":
            d, i, s = (self.ssa[o] for o in op.operands)
            td, ti, ts = (self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            self.lines.append(
                f'{indent}{r} = intent.scatter {d}, {i}, {s} '
                f'{{axis = {axis} : i64, mode = "{a.get("mode", "along_axis")}", '
                f'reduce = "{a.get("reduce", "none")}"}} : '
                f'({td}, {ti}, {ts}) -> {self._ttn(op.result, env)}')
        elif op.op == "nonzero":
            # Dynamic-shape output (C8): [nnz, ndim] i64 index rows. The emitted
            # leading extent is the MAX upper bound (prod of cond shape).
            c = self.ssa[op.operands[0]]
            self.int_results[op.result] = "i64"
            self.lines.append(
                f'{indent}{r} = intent.nonzero {c} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "masked_select":
            d, m = (self.ssa[o] for o in op.operands)
            td, tm = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.masked_select {d}, {m} : '
                f'({td}, {tm}) -> {self._ttn(op.result, env)}')
        elif op.op == "masked_scatter":
            d, m, s = (self.ssa[o] for o in op.operands)
            td, tm, ts = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.masked_scatter {d}, {m}, {s} : '
                f'({td}, {tm}, {ts}) -> {self._ttn(op.result, env)}')
        elif op.op == "concat":
            xs = ", ".join(self.ssa[o] for o in op.operands)
            txs = ", ".join(self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            self.lines.append(
                f'{indent}{r} = intent.concat {xs} {{axis = {axis} : i64}} : '
                f'({txs}) -> {self._ttn(op.result, env)}')
        elif op.op == "pad":
            x = self.ssa[op.operands[0]]
            lo = ", ".join(str(int(v)) for v in a["lo"][fold:])
            hi = ", ".join(str(int(v)) for v in a["hi"][fold:])
            self.lines.append(
                f'{indent}{r} = intent.pad {x} {{lo = [{lo}], hi = [{hi}], '
                f'value = {_mlir_f(a.get("value", 0.0))} : f64}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "tile":
            x = self.ssa[op.operands[0]]
            reps = ", ".join(str(int(v)) for v in a["reps"][fold:])
            self.lines.append(
                f'{indent}{r} = intent.tile {x} {{reps = [{reps}]}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "flip":
            x = self.ssa[op.operands[0]]
            axes = ", ".join(str(int(v) - fold) for v in a["axes"])
            self.lines.append(
                f'{indent}{r} = intent.flip {x} {{axes = [{axes}]}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "triu":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.triu {x} {{k = {int(a.get("k", 0))} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "iota":
            # Integer column-index vector; the result is an integer tensor (not
            # f32), so record it as a non-f32 op and print its int type directly.
            dt = a.get("dtype", "i32")
            n = int(a["n"])
            self.int_results[op.result] = dt
            self.lines.append(
                f'{indent}{r} = intent.iota : tensor<{n}x{dt}>')
        elif op.op == "rand":
            # Counter-based uniform[0,1) RNG (C5); no operands, shape+dtype in the
            # result type, seed in the attr.
            self.lines.append(
                f'{indent}{r} = intent.rand {{seed = {int(a["seed"])} : i64}} : '
                f'{_tt(env[op.result])}')
        elif op.op == "const":
            # A literal value -> standard arith.constant (no intent semantics to
            # lower). A scalar splats dense over the (possibly rank-0) result
            # tensor; a list value (iota/arange materialized as an explicit
            # vector) prints as a dense elements array. An i1/int-dtyped const
            # (a boolean sentinel / mask threshold) carries its own element type.
            val = a["value"]
            dt = a.get("dtype", "f32")
            if dt in ("i1", "bool"):
                self.bool_results.add(op.result)
                truthy = val not in (0, 0.0, False, None)
                dense = f"dense<{'true' if truthy else 'false'}>"
            elif dt in ("i8", "i16", "i32", "i64"):
                self.int_results[op.result] = dt
                if isinstance(val, (list, tuple)):
                    dense = "dense<[" + ", ".join(str(int(v)) for v in val) + "]>"
                else:
                    dense = f"dense<{int(val)}>"
            elif isinstance(val, (list, tuple)):
                dense = "dense<[" + ", ".join(_mlir_f(v) for v in val) + "]>"
            else:
                dense = f"dense<{_mlir_f(val)}>"
            self.lines.append(
                f'{indent}{r} = arith.constant {dense} : {self._ttn(op.result, env)}')
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
        _infer_body(loop.body, env, fold, extents)
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
    _infer_body(fn.body, env, fold, extents)

    p = _Printer()
    # Integer inputs (e.g. a gather/scatter `index` tensor) keep their declared
    # integer width; every other input is emitted in the execute dtype (f32).
    for t in fn.inputs:
        if t.dtype in ("i8", "i16", "i32", "i64"):
            p.int_results[t.name] = t.dtype
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
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
    # Respect i1 (comparison) / integer (iota) result element types so the
    # function signature matches the value actually produced.
    otys = ", ".join(p._ttn(n, env) for n in outs)
    name = func_name or fn.name

    head = (f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
            f"(flat graph, fold={fold}).\n"
            f"module {{\n"
            f"  func.func @{name}({', '.join(args)}) -> ({otys}) {{")
    tail = f"    return {ovals} : {otys}\n  }}\n}}\n"
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


__all__ = ["emit_body_function", "emit_loop_module", "emit_module"]
