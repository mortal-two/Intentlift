"""IntentIR reference interpreter (executable value semantics).

This *implements* the written reference semantics (spec/ir/) — it is NOT itself
the definition. Each op is a thin numpy denotation to keep the trusted computing
base small. Float note: numpy ``sum`` is pairwise and einsum dispatches to BLAS,
so the interpreter commits to numpy's order and float agreement is ALWAYS
tolerance-based (even against the interpreter itself) — never advertise "exact".

Layer A only is computed. Layer B (e.g. ``assoc``) parameterizes downstream
tolerance; Layer C is ignored here.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from intentir.ir.model import IntentFunction, IntentOp

_DTYPE = {
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
    "bf16": np.float32,  # numpy has no bf16; approximate as f32 for the reference
    "i1": np.bool_, "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
}

_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "max": np.maximum, "min": np.minimum, "pow": np.power,
    # atan2(y, x): the complex angle primitive (angle = atan2(imag, real)).
    "atan2": np.arctan2,
    # Comparison fns return an i1 (bool) array (the condition for `select`).
    "gt": np.greater, "ge": np.greater_equal, "lt": np.less,
    "le": np.less_equal, "eq": np.equal, "ne": np.not_equal,
}

try:  # accurate double-precision erf reference; fall back to the stdlib scalar.
    from scipy.special import erf as _erf  # type: ignore
except Exception:  # pragma: no cover - scipy usually present
    import math as _math
    _erf = np.vectorize(_math.erf)

_UNARY = {
    "identity": lambda x: x,
    "exp": np.exp, "log": np.log, "log2": np.log2, "floor": np.floor,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: np.reciprocal(np.sqrt(x)), "neg": np.negative,
    "recip": np.reciprocal, "abs": np.abs, "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "cos": np.cos, "sin": np.sin, "tan": np.tan, "exp2": np.exp2,
    "atan": np.arctan,
    "erf": _erf, "isnan": np.isnan, "logical_not": np.logical_not,
}

_REDUCE = {
    "sum": np.sum, "max": np.max, "min": np.min, "prod": np.prod,
    "mean": np.mean, "any": np.any, "all": np.all,
}


class InterpError(Exception):
    pass


# --- philox4x32-10 counter-based RNG (matches the C++ convert-intent-rng-to-gpu
# lowering bit-for-bit up to f32 rounding). value[i] = philox(seed, i) -> two
# independent uniforms in [0,1). Public philox constants; 10 rounds. ------------
_PHILOX_M0 = np.uint32(0xD2511F53)
_PHILOX_M1 = np.uint32(0xCD9E8D57)
_PHILOX_W0 = np.uint32(0x9E3779B9)
_PHILOX_W1 = np.uint32(0xBB67AE85)


def _philox_uniform(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two length-`n` uniform[0,1) f32 arrays; element i hashes (seed, i)."""
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    c0 = np.arange(n, dtype=np.uint32)
    c1 = np.zeros(n, dtype=np.uint32)
    c2 = np.zeros(n, dtype=np.uint32)
    c3 = np.zeros(n, dtype=np.uint32)
    k0 = np.full(n, np.uint32(seed & 0xFFFFFFFF))
    k1 = np.full(n, np.uint32((seed >> 32) & 0xFFFFFFFF))
    with np.errstate(over="ignore"):
        for r in range(10):
            p0 = c0.astype(np.uint64) * np.uint64(_PHILOX_M0)
            hi0 = (p0 >> np.uint64(32)).astype(np.uint32)
            lo0 = p0.astype(np.uint32)
            p1 = c2.astype(np.uint64) * np.uint64(_PHILOX_M1)
            hi1 = (p1 >> np.uint64(32)).astype(np.uint32)
            lo1 = p1.astype(np.uint32)
            c0, c1, c2, c3 = (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0
            if r < 9:
                k0 = k0 + _PHILOX_W0
                k1 = k1 + _PHILOX_W1

    def unit(c: np.ndarray) -> np.ndarray:
        # top 24 bits -> [0,1) f32 (mantissa-exact, matches uitofp(c>>8)*2^-24).
        return (c >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)

    return unit(c0), unit(c1)


def _eval_op(op: IntentOp, env: dict[str, np.ndarray],
             dims: dict[str, int] | None = None) -> np.ndarray:
    o = op.op
    a = op.attrs
    dims = dims or {}

    def _dim(s: Any) -> int:  # resolve an int-or-symbolic shape entry
        return int(dims[s]) if isinstance(s, str) else int(s)

    if o == "const":
        # A literal value (scalar or filled tensor). Not "intent" to be lowered;
        # the emitter turns it into arith.constant. shape [] -> scalar.
        shape = tuple(_dim(s) for s in a.get("shape", []))
        val = np.full(shape, a["value"], dtype=_DTYPE.get(a.get("dtype", "f32")))
        return val if shape else _DTYPE.get(a.get("dtype", "f32"))(a["value"])

    if o == "map_binary":
        fn = a["fn"]
        if fn not in _BINARY:
            raise InterpError(f"map_binary: unknown fn {fn!r}")
        lhs, rhs = env[op.operands[0]], env[op.operands[1]]
        return _BINARY[fn](lhs, rhs)  # numpy broadcasting realizes Layer C 'broadcast'

    if o == "map_unary":
        fn = a["fn"]
        if fn not in _UNARY:
            raise InterpError(f"map_unary: unknown fn {fn!r}")
        return np.asarray(_UNARY[fn](env[op.operands[0]]))

    if o == "reduce":
        fn = a["fn"]
        if fn not in _REDUCE:
            raise InterpError(f"reduce: unknown fn {fn!r}")
        axes = tuple(a.get("axes", []))
        # keepdims=True so the reduced result stays broadcastable (rowmax/rowsum).
        return _REDUCE[fn](env[op.operands[0]], axis=axes, keepdims=True)

    if o == "scan":
        fn = a["fn"]
        x, ax = env[op.operands[0]], int(a["axis"])
        if fn == "sum":
            return np.cumsum(x, axis=ax)
        if fn == "max":  # cummax value (torch.cummax also returns indices)
            return np.maximum.accumulate(x, axis=ax)
        if fn == "min":  # cummin value
            return np.minimum.accumulate(x, axis=ax)
        raise InterpError(f"scan: unknown fn {fn!r}")

    if o == "arg_reduce":
        # Index of the extremum along `axis` (torch.argmax/argmin). Ties -> the
        # first/lowest index, which is exactly numpy argmax/argmin's contract.
        fn = a["fn"]
        x, ax = env[op.operands[0]], int(a["axis"])
        if fn == "argmax":
            idx = np.argmax(x, axis=ax)
        elif fn == "argmin":
            idx = np.argmin(x, axis=ax)
        else:
            raise InterpError(f"arg_reduce: unknown fn {fn!r}")
        return idx.astype(np.int32)

    if o == "sort":
        # Full sort along the last axis (C6 HARD tier). Produces sorted VALUES
        # and the argsort INDICES (torch.sort / numpy.sort+argsort). `descending`
        # flips the direction; a result `axis` extent W < N keeps the first W
        # (topk). `op.result` is a list [values, indices] when both are named, or
        # a single str for a values-only / indices-only op. A stable argsort gives
        # ONE valid permutation; the lowering's bitonic ties are unstable but
        # reproduce the same sorted values.
        x, ax = env[op.operands[0]], int(a.get("axis", -1))
        desc = bool(a.get("descending", False))
        idx = np.argsort(x, axis=ax, kind="stable")
        if desc:
            idx = np.flip(idx, axis=ax)
        vals = np.take_along_axis(x, idx, axis=ax)
        w = a.get("k")  # optional topk width
        if w is not None:
            vals = np.take(vals, np.arange(int(w)), axis=ax)
            idx = np.take(idx, np.arange(int(w)), axis=ax)
        vals = vals.astype(x.dtype)
        idx = idx.astype(np.int32)
        if isinstance(op.result, list):
            return (vals, idx)
        # single result: values unless the op is explicitly index-typed (argsort).
        return idx if a.get("return") == "indices" else vals

    if o == "contraction":
        sub = a["subscripts"]
        return np.einsum(sub, env[op.operands[0]], env[op.operands[1]])

    if o == "iota":
        # Column-index vector [0, 1, ..., n-1] (numpy arange). `n` is the static
        # extent; the optional `dtype` selects the integer width (default i32).
        n = int(a["n"])
        return np.arange(n, dtype=_DTYPE.get(a.get("dtype", "i32"), np.int32))

    if o == "select":
        c, x, y = (env[n] for n in op.operands)
        return np.where(c, x, y)

    if o == "cast":
        return env[op.operands[0]].astype(_DTYPE[a["to_dtype"]])

    if o == "rand":
        # Counter-based uniform[0,1) (C5). Order-independent: element i = the
        # philox hash of (seed, i), reshaped to the result's shape.
        shape = tuple(_dim(s) for s in a.get("shape", []))
        n = int(np.prod(shape)) if shape else 1
        u0, _u1 = _philox_uniform(int(a["seed"]), n)
        return u0.reshape(shape) if shape else np.float32(u0[0])

    if o == "gather":
        # Data-dependent addressing (C3). "along_axis" = torch.gather /
        # numpy take_along_axis (index shaped like the result); "take" =
        # index_select / embedding row-gather = numpy take along `axis`.
        data = env[op.operands[0]]
        index = np.asarray(env[op.operands[1]]).astype(np.int64)
        axis = int(a["axis"])
        mode = a.get("mode", "along_axis")
        if mode == "along_axis":
            return np.take_along_axis(data, index, axis=axis)
        if mode == "take":
            return np.take(data, index, axis=axis)
        raise InterpError(f"gather: unknown mode {mode!r}")

    if o == "scatter":
        # In-place-semantics scatter into a COPY of data (torch.scatter /
        # scatter_add_). reduce "none" overwrites (numpy put_along_axis);
        # "add" accumulates order-independently (numpy add.at along the axis).
        data = env[op.operands[0]]
        index = np.asarray(env[op.operands[1]]).astype(np.int64)
        src = env[op.operands[2]]
        axis = int(a["axis"])
        reduce = a.get("reduce", "none")
        out = np.array(data, copy=True)
        if reduce == "add":
            grid = list(np.indices(index.shape))
            grid[axis] = index
            np.add.at(out, tuple(grid), src)
        elif reduce == "none":
            np.put_along_axis(out, index, src, axis=axis)
        else:
            raise InterpError(f"scatter: unknown reduce {reduce!r}")
        return out

    if o == "nonzero":
        # Dynamic-shape output (C8): row-major multi-indices of the nonzero
        # elements (torch.nonzero / numpy.argwhere). Result [nnz, ndim] i64; the
        # leading extent nnz is data-dependent. np.argwhere is C-order, matching
        # torch.nonzero's row-major scan.
        cond = env[op.operands[0]]
        return np.argwhere(cond != 0).astype(np.int64)

    if o == "masked_select":
        # Dynamic-shape output (C8): elements where mask is true, flattened in
        # row-major order (torch.masked_select / data[mask]). Result [nnz].
        data = env[op.operands[0]]
        mask = env[op.operands[1]] != 0
        return data[mask]

    if o == "masked_scatter":
        # C8 compaction primitive with a data-shaped output: overwrite each
        # masked position with the next src element in row-major order
        # (torch.masked_scatter). Boolean assignment on a C-contiguous array
        # consumes src[0..nnz) in row-major order.
        data = env[op.operands[0]]
        mask = env[op.operands[1]] != 0
        src = np.asarray(env[op.operands[2]]).reshape(-1)
        out = np.array(data, copy=True)
        n = int(mask.sum())
        out[mask] = src[:n].astype(out.dtype)
        return out

    if o == "concat":
        # Structural layout (C4). Variadic inputs placed end-to-end along `axis`
        # at their cumulative offsets (numpy concatenate / torch.cat; hstack /
        # vstack = concat on axis 1/0).
        return np.concatenate([env[n] for n in op.operands], axis=int(a["axis"]))

    if o == "pad":
        # Constant pad `lo`/`hi` elements before/after each dim, border = `value`
        # (numpy pad mode="constant" / torch constant_pad_nd).
        lo = [int(x) for x in a["lo"]]
        hi = [int(x) for x in a["hi"]]
        return np.pad(env[op.operands[0]], list(zip(lo, hi)), mode="constant",
                      constant_values=a.get("value", 0.0))

    if o == "tile":
        # Repeat the whole tensor `reps[d]` times along each dim (numpy tile /
        # torch.tile / torch.Tensor.repeat).
        return np.tile(env[op.operands[0]], [int(x) for x in a["reps"]])

    if o == "flip":
        # Reverse the element order along each axis in `axes` (numpy flip /
        # torch.flip).
        return np.flip(env[op.operands[0]], axis=tuple(int(x) for x in a["axes"]))

    if o == "triu":
        # Upper triangle above the k-th diagonal, rest zeroed (numpy/torch triu).
        return np.triu(env[op.operands[0]], k=int(a.get("k", 0)))

    if o == "pool":
        # Windowed pooling (C7). avg = window mean with torch's default
        # count_include_pad=True (divisor = kH*kW); max = window max + (for
        # max_pool2d_with_indices) the argmax FLAT index ih*W+iw within the (H,W)
        # plane, first-index tie-break (strict `>` scan in kh,kw order).
        x = env[op.operands[0]]
        fn = a["fn"]
        kH, kW = int(a["kH"]), int(a["kW"])
        sH, sW = int(a["sH"]), int(a["sW"])
        padH, padW = int(a["padH"]), int(a["padW"])
        N, C, H, W = x.shape
        OH = (H + 2 * padH - kH) // sH + 1
        OW = (W + 2 * padW - kW) // sW + 1
        sumacc = np.zeros((N, C, OH, OW), np.float32)
        maxacc = np.full((N, C, OH, OW), -np.inf, np.float32)
        idxacc = np.full((N, C, OH, OW), -1, np.int64)
        oh_i = np.arange(OH)
        ow_i = np.arange(OW)
        for kh in range(kH):
            ih = oh_i * sH - padH + kh                    # (OH,)
            hvalid = (ih >= 0) & (ih < H)
            ihc = np.clip(ih, 0, H - 1)
            for kw in range(kW):
                iw = ow_i * sW - padW + kw                # (OW,)
                wvalid = (iw >= 0) & (iw < W)
                iwc = np.clip(iw, 0, W - 1)
                valid = hvalid[:, None] & wvalid[None, :]  # (OH,OW)
                patch = x[:, :, ihc[:, None], iwc[None, :]]  # (N,C,OH,OW)
                if fn == "avg":
                    sumacc += np.where(valid, patch, 0.0)
                elif fn == "max":
                    flat = ih[:, None] * W + iw[None, :]   # (OH,OW)
                    better = valid[None, None] & (patch > maxacc)
                    maxacc = np.where(better, patch, maxacc)
                    idxacc = np.where(better,
                                      np.broadcast_to(flat, (N, C, OH, OW)), idxacc)
                else:
                    raise InterpError(f"pool: unknown fn {fn!r}")
        if fn == "avg":
            return (sumacc / np.float32(kH * kW)).astype(x.dtype)
        vals = maxacc.astype(x.dtype)
        if isinstance(op.result, list):     # max_pool2d_with_indices
            return (vals, idxacc.astype(np.int64))
        return vals

    if o == "upsample":
        # Nearest-neighbor resize (C7): out[.,oh,ow] = in[.,floor(oh*IH/OH),
        # floor(ow*IW/OW)] with reciprocal scale IH/OH (matches flag_gems /
        # torch.upsample_nearest2d). `scale` = [sh, sw] integer upscale factors.
        x = env[op.operands[0]]
        N, C, H, W = x.shape
        sh, sw = (int(v) for v in a["scale"])
        OH, OW = H * sh, W * sw
        ih = np.minimum((np.arange(OH) * (H / OH)).astype(np.int64), H - 1)
        iw = np.minimum((np.arange(OW) * (W / OW)).astype(np.int64), W - 1)
        return x[:, :, ih[:, None], iw[None, :]]

    if o == "polar":
        # Complex build (C7): real = abs*cos(angle), imag = abs*sin(angle). The
        # complex result is the trailing real/imag PAIR (two real results).
        absv, ang = env[op.operands[0]], env[op.operands[1]]
        return (absv * np.cos(ang), absv * np.sin(ang))

    if o == "conv":
        # Forward conv1d/2d/3d (C7): cross-correlation (no kernel flip, matching
        # torch.nn.functional.conv{1,2,3}d). input [N,Cin,*spatial], weight
        # [Cout,Cin/groups,*kernel], optional bias [Cout] -> [N,Cout,*out]. A
        # direct windowed-weighted accumulation over (input channel, kernel tap),
        # zero-padded, grouped. Verified rel<1e-3 (fp accumulation order).
        x = env[op.operands[0]].astype(np.float32)
        w = env[op.operands[1]].astype(np.float32)
        bias = env[op.operands[2]].astype(np.float32) if len(op.operands) > 2 \
            else None
        R = x.ndim - 2
        stride = [int(v) for v in a["stride"]]
        pad = [int(v) for v in a["padding"]]
        dil = [int(v) for v in a["dilation"]]
        groups = int(a.get("groups", 1))
        N, Cin = x.shape[0], x.shape[1]
        Cout = w.shape[0]
        ipg, opg = Cin // groups, Cout // groups
        Hs, Ks = x.shape[2:], w.shape[2:]
        Os = [(Hs[j] + 2 * pad[j] - dil[j] * (Ks[j] - 1) - 1) // stride[j] + 1
              for j in range(R)]
        xp = np.pad(x, [(0, 0), (0, 0)] + [(pad[j], pad[j]) for j in range(R)])
        out = np.zeros((N, Cout) + tuple(Os), np.float32)
        for kk in itertools.product(*[range(k) for k in Ks]):
            slc = [slice(None), slice(None)]
            for j in range(R):
                start = kk[j] * dil[j]
                slc.append(slice(start, start + stride[j] * (Os[j] - 1) + 1,
                                 stride[j]))
            xs = xp[tuple(slc)]                          # [N, Cin, *O]
            wk = w[(slice(None), slice(None)) + kk]      # [Cout, ipg]
            for g in range(groups):
                xg = xs[:, g * ipg:(g + 1) * ipg]        # [N, ipg, *O]
                wg = wk[g * opg:(g + 1) * opg]           # [opg, ipg]
                # tensordot over ipg -> [N, *O, opg]; move opg to channel axis.
                contrib = np.tensordot(xg, wg, axes=([1], [1]))
                out[:, g * opg:(g + 1) * opg] += np.moveaxis(contrib, -1, 1)
        if bias is not None:
            out += bias.reshape((1, Cout) + (1,) * R)
        return out

    if o == "transpose":
        return np.transpose(env[op.operands[0]], axes=a.get("perm"))

    raise InterpError(f"interpreter: op {o!r} not implemented")


# Carry-init keywords (structured loop). Scalars broadcast against the per-row
# carry shapes, so the empty-accumulator first iteration is handled by alpha=0
# (exp(-inf - m_blk)) with no explicit shape inference needed.
_INIT = {"NEG_INF": -np.inf, "zeros": 0.0, "ZERO": 0.0, "ones": 1.0}


def _carry_init(spec: dict[str, Any]) -> np.ndarray:
    v = spec["init"]
    if isinstance(v, str):
        if v in _INIT:
            return np.asarray(_INIT[v], dtype=np.float32)
        return np.asarray(float(v), dtype=np.float32)  # numeric literal as string
    return np.asarray(v, dtype=np.float32)


def _run_body(
    ops: list[IntentOp], env: dict[str, Any], params: dict[str, int]
) -> dict[str, Any]:
    env = dict(env)
    for op in ops:
        if op.op == "loop":
            env = _eval_loop(op, env, params)
            continue
        out = _eval_op(op, env, params)
        if isinstance(op.result, list):  # multi-result (e.g. sort -> values,indices)
            for name, val in zip(op.result, out):
                env[name] = val
        elif op.result is not None:
            env[op.result] = out  # non-loop result is always a str
    return env


def _eval_loop(
    op: IntentOp, env: dict[str, Any], params: dict[str, int]
) -> dict[str, Any]:
    """Structured loop with carry (online-softmax recurrence).

    Iterates the complete semantic trip axis as one logical step, slicing each
    ``scan`` source, running the body with carries + sliced inputs, then
    advancing carries from ``yield``. Source tiling is deliberately unavailable
    here: it belongs to C/pi, not to the Layer-A loop.
    """
    carries = {c["name"]: _carry_init(c) for c in op.carry}

    if not op.scan:
        raise InterpError("loop without scan inputs is not supported")
    src0 = op.scan[0]
    extent = env[src0["from"]].shape[src0["axis"]]

    tile = extent

    j = 0
    while j < extent:
        j1 = min(j + tile, extent)
        sub: dict[str, Any] = dict(env)
        sub.update(carries)
        for s in op.scan:
            arr = env[s["from"]]
            idx: list[Any] = [slice(None)] * arr.ndim
            idx[s["axis"]] = slice(j, j1)
            sub[s["name"]] = arr[tuple(idx)]
        sub = _run_body(op.body, sub, params)
        carries = {c["name"]: sub[y] for c, y in zip(op.carry, op.yields)}
        j = j1

    out = dict(env)
    for name, c in zip(op.result, op.carry):
        out[name] = carries[c["name"]]
    return out


def interpret(
    fn: IntentFunction,
    inputs: dict[str, np.ndarray],
    params: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Execute an IntentFunction, returning {output_name: ndarray}.

    ``params`` supplies concrete symbolic tensor extents. Loop tile choices are
    intentionally not accepted by the Layer-A interpreter.
    """
    # Resolve symbolic shape dims (e.g. a `rand`/`const` shape ["M","N"]) from the
    # concrete input arrays, merged with any caller-supplied semantic extents.
    dims: dict[str, int] = dict(params or {})
    for t in fn.inputs:
        arr = inputs.get(t.name)
        if arr is not None and hasattr(arr, "shape"):
            for s, sz in zip(t.shape, arr.shape):
                if isinstance(s, str):
                    dims[s] = int(sz)
    env = _run_body(fn.body, dict(inputs), dims)
    missing = [t.name for t in fn.outputs if t.name not in env]
    if missing:
        raise InterpError(f"outputs not produced: {missing}")
    return {t.name: env[t.name] for t in fn.outputs}
