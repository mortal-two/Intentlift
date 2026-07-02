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

from typing import Any

import numpy as np

from ..ir.model import IntentFunction, IntentOp

_DTYPE = {
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
    "bf16": np.float32,  # numpy has no bf16; approximate as f32 for the reference
    "i1": np.bool_, "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
}

_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "max": np.maximum, "min": np.minimum, "pow": np.power,
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
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "neg": np.negative,
    "recip": np.reciprocal, "abs": np.abs, "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "cos": np.cos, "sin": np.sin, "tan": np.tan, "exp2": np.exp2,
    "erf": _erf, "isnan": np.isnan, "logical_not": np.logical_not,
}

_REDUCE = {
    "sum": np.sum, "max": np.max, "min": np.min, "prod": np.prod,
    "mean": np.mean, "any": np.any, "all": np.all,
}


class InterpError(Exception):
    pass


def _eval_op(op: IntentOp, env: dict[str, np.ndarray]) -> np.ndarray:
    o = op.op
    a = op.attrs

    if o == "const":
        # A literal value (scalar or filled tensor). Not "intent" to be lowered;
        # the emitter turns it into arith.constant. shape [] -> scalar.
        shape = tuple(a.get("shape", []))
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
        if fn != "sum":
            raise InterpError(f"scan: unknown fn {fn!r}")
        return np.cumsum(env[op.operands[0]], axis=int(a["axis"]))

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
        out = _eval_op(op, env)
        if op.result is not None:
            env[op.result] = out  # non-loop result is always a str
        if op.out is not None:
            env[op.out] = out
    return env


def _eval_loop(
    op: IntentOp, env: dict[str, Any], params: dict[str, int]
) -> dict[str, Any]:
    """Structured loop with carry (online-softmax recurrence).

    Iterates the trip axis in tiles, slicing each ``scan`` source for the tile,
    running the body with carries + sliced inputs, then advancing carries from
    ``yield``. The tile size comes from ``params[trip.tile_hint]``; absent that,
    the whole axis runs as one iteration (mathematically the naive form). Online
    softmax is exact (up to fp) for any tiling, so tile size is a Layer-C choice.
    """
    carries = {c["name"]: _carry_init(c) for c in op.carry}

    if not op.scan:
        raise InterpError("loop without scan inputs is not supported")
    src0 = op.scan[0]
    extent = env[src0["from"]].shape[src0["axis"]]

    tile = None
    if op.trip is not None:
        tile = params.get(op.trip.get("tile_hint"))
    if tile is None:
        tile = extent  # single iteration

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

    ``params`` supplies concrete values for symbolic loop tile hints (e.g.
    {"BK": 32}); omit to run loops as a single full-extent iteration.
    """
    env = _run_body(fn.body, dict(inputs), params or {})
    missing = [t.name for t in fn.outputs if t.name not in env]
    if missing:
        raise InterpError(f"outputs not produced: {missing}")
    return {t.name: env[t.name] for t in fn.outputs}
