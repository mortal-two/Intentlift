"""Emit `intent`-dialect MLIR from an IntentIR JSON function.

This is the ``JSON -> intent MLIR dialect`` front-end leg. It walks the
:class:`~intentir.ir.model.IntentFunction` body generically (it is driven by the
op list, not hard-coded to FlashAttention) and prints `intent`-dialect text at
concrete shapes, doing einsum/broadcast/reduce shape inference that matches the
archived NumPy interpreter (``intentir.verify.back.legacy_v0.interpret``).

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
executable form). Scalar inputs (``scale``) are emitted as rank-0 tensors while
retaining their declared element type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ast
import math
import operator
import re
from typing import AbstractSet

from intentir.ir.model import IntentFunction, IntentOp

IndexBoundsProjection = frozenset[str]

_INVALID_MLIR_SYMBOL_CHAR = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_mlir_symbol(name: str) -> str:
    """Return a deterministic bare MLIR symbol derived from ``name``.

    The emitted symbol is deliberately stricter than MLIR's bare-symbol grammar:
    NVPTX reuses it as a PTX ``.entry`` name, where punctuation such as ``.`` or
    ``$`` can make ptxas reject otherwise valid MLIR.  Keep ASCII letters,
    digits and ``_`` only, then add a leading ``_`` when needed (including the
    empty name).
    """

    symbol = _INVALID_MLIR_SYMBOL_CHAR.sub("_", name)
    if not symbol or not re.match(r"[A-Za-z_]", symbol[0]):
        symbol = "_" + symbol
    return symbol


def _resolve(shape, extents) -> list[int]:
    return [int(extents[s]) if isinstance(s, str) else int(s) for s in shape]


_INT_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _resolve_int(value, extents: dict[str, int] | None = None) -> int:
    """Resolve one trusted integer shape expression without Python ``eval``."""

    if isinstance(value, bool):
        raise TypeError("boolean is not an integer extent")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return int(value)
    bindings = dict(extents or {})
    tree = ast.parse(value, mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id not in bindings:
                raise ValueError(
                    f"unknown symbolic integer extent {node.id!r} in {value!r}"
                )
            return int(bindings[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and type(node.op) in _INT_BINOPS:
            lhs, rhs = visit(node.left), visit(node.right)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and rhs == 0:
                raise ValueError(
                    f"zero divisor in integer extent expression {value!r}"
                )
            return int(_INT_BINOPS[type(node.op)](lhs, rhs))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ceil_div"
            and len(node.args) == 2
            and not node.keywords
        ):
            lhs, rhs = visit(node.args[0]), visit(node.args[1])
            if lhs < 0 or rhs <= 0:
                raise ValueError(
                    "ceil_div shape expressions require a non-negative "
                    f"numerator and positive denominator: {value!r}"
                )
            return (lhs + rhs - 1) // rhs
        raise ValueError(f"unsupported integer extent expression {value!r}")

    resolved = visit(tree)
    if resolved < _I64_MIN or resolved > _I64_MAX:
        raise ValueError(f"integer extent expression is outside int64: {value!r}")
    return resolved


def _fold_permutation(values, fold: int, extents) -> list[int]:
    perm = [_resolve_int(value, extents) for value in values]
    if fold == 0:
        return perm
    if perm[:fold] != list(range(fold)) or any(
        value < fold for value in perm[fold:]
    ):
        raise ValueError(
            "emit_mlir: cannot fold batch dimensions moved by transpose"
        )
    return [value - fold for value in perm[fold:]]


def _tt(shape: list[int], dtype: str) -> str:
    """MLIR ranked-tensor type for a (possibly rank-0) shape."""
    if not shape:
        return f"tensor<{dtype}>"
    return "tensor<" + "x".join(str(s) for s in shape) + f"x{dtype}>"


def _strip(group: str, fold: int) -> str:
    """Drop the leading `fold` (batch) labels from one einsum label group."""
    return group[fold:]


BroadcastAxes = Mapping[str, Mapping[str, Sequence[int]]]


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


def _broadcast_checked(
    a: list[int] | None,
    b: list[int] | None,
) -> list[int] | None:
    """Strict numpy broadcast used when J supplies explicit broadcast facts."""

    if a is None:
        return b
    if b is None:
        return a
    rank = max(len(a), len(b))
    lhs = [1] * (rank - len(a)) + list(a)
    rhs = [1] * (rank - len(b)) + list(b)
    result: list[int] = []
    for left, right in zip(lhs, rhs, strict=True):
        if left != right and left != 1 and right != 1:
            raise ValueError(
                f"explicit broadcast shapes are incompatible: {a} vs {b}"
            )
        result.append(max(left, right))
    return result


def _broadcast_projection(
    shape: list[object] | None,
    axes: Sequence[int],
    target_shape: Sequence[object] | None = None,
    *,
    unit_dimensions: AbstractSet[str] = frozenset(),
) -> tuple[list[object] | None, tuple[int, ...]]:
    """Project one operand onto the consumer rank using Layer-B axes.

    A broadcast axis may already exist as a size-one dimension (``[M, 1]``
    broadcast over axis 1) or may be absent from a lower-rank operand
    (``[N]`` broadcast over axis 0 into ``[M, N]``).  Only the latter case
    requires ``tensor.expand_shape`` rank insertion.  When a consumer shape is
    available, preserve the operand dimensions in order and allow dimensions
    to be omitted only at declared broadcast axes.
    """

    if shape is None:
        return None, ()
    if not axes:
        return list(shape), ()
    target_rank = (
        len(target_shape)
        if target_shape is not None
        else len(shape) + len(axes)
    )
    normalized: list[int] = []
    for raw_axis in axes:
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
            raise NotImplementedError(
                "emit_mlir broadcast axes currently require integer positions"
            )
        axis = raw_axis if raw_axis >= 0 else raw_axis + target_rank
        if axis < 0 or axis >= target_rank:
            raise ValueError(
                f"broadcast axis {raw_axis} is outside target rank {target_rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ValueError("broadcast axes must be unique")

    if target_shape is None:
        inserted = set(normalized)
        source = iter(shape)
        return (
            [
                1 if index in inserted else next(source)
                for index in range(target_rank)
            ],
            tuple(sorted(inserted)),
        )

    target = list(target_shape)
    if len(shape) > target_rank:
        raise ValueError(
            f"broadcast axes {list(axes)} cannot project rank {len(shape)} "
            f"operand to consumer rank {target_rank}"
        )
    if not shape:
        # A rank-0 tensor uses the empty affine input map and therefore
        # broadcasts across the complete consumer domain without materializing
        # rank insertion.  Boundary axes remain audited semantic facts, but
        # need not enumerate every consumer dimension for a scalar.
        return [1] * target_rank, tuple(range(target_rank))
    axis_set = set(normalized)
    memo: dict[tuple[int, int], tuple[int, ...] | None] = {}

    def is_unit(extent: object) -> bool:
        return extent == 1 or (
            isinstance(extent, str) and extent in unit_dimensions
        )

    def same_extent(left: object, right: object) -> bool:
        return left == right or (is_unit(left) and is_unit(right))

    def align(source_index: int, target_index: int) -> tuple[int, ...] | None:
        key = (source_index, target_index)
        if key in memo:
            return memo[key]
        if target_index == target_rank:
            result = () if source_index == len(shape) else None
            memo[key] = result
            return result

        if target_index in axis_set:
            # Prefer retaining an already-present singleton at the same target
            # position.  If that cannot complete the alignment, treat this
            # declared broadcast axis as the missing rank dimension.
            if source_index < len(shape) and is_unit(shape[source_index]):
                suffix = align(source_index + 1, target_index + 1)
                if suffix is not None:
                    memo[key] = suffix
                    return suffix
            suffix = align(source_index, target_index + 1)
            if suffix is not None:
                result = (target_index, *suffix)
                memo[key] = result
                return result
        elif (
            source_index < len(shape)
            and same_extent(shape[source_index], target[target_index])
        ):
            suffix = align(source_index + 1, target_index + 1)
            if suffix is not None:
                memo[key] = suffix
                return suffix

        memo[key] = None
        return None

    inserted_axes = align(0, 0)
    if inserted_axes is None:
        raise ValueError(
            f"operand shape {shape} cannot align with consumer shape {target} "
            f"using broadcast axes {list(axes)}; existing broadcast "
            "dimensions must be size one and missing dimensions must be "
            "declared broadcast axes"
        )
    projected = [
        1 if index in axis_set else target_extent
        for index, target_extent in enumerate(target)
    ]
    return projected, inserted_axes


def _insert_broadcast_axes(
    shape: list[int] | None,
    axes: Sequence[int],
    target_shape: Sequence[int] | None = None,
) -> list[int] | None:
    projected, _inserted_axes = _broadcast_projection(
        shape, axes, target_shape
    )
    return projected


def _op_broadcast_axes(
    op: IntentOp,
    broadcast_axes: BroadcastAxes | None,
) -> Mapping[str, Sequence[int]]:
    if broadcast_axes is None or not isinstance(op.result, str):
        return {}
    return broadcast_axes.get(op.result, {})


def _broadcast_mul_contraction_subscripts(
    op: IntentOp,
    env: Mapping[str, list[int]],
    axes: Mapping[str, Sequence[int]],
) -> str | None:
    """Canonicalize a non-trailing rank-insertion multiply to einsum.

    ``map_binary`` plus Layer-B broadcast axes is the canonical JSON spelling,
    while the row CUDA pass also accepts the equivalent no-reduction
    contraction.  Converting only multiplication with a proven non-trailing
    insertion keeps the transformation exact and avoids forcing live LLMs to
    choose one backend-facing graph spelling.  Ordinary trailing vector
    broadcasts remain map operations.
    """

    if (
        op.op != "map_binary"
        or op.attrs.get("fn") != "mul"
        or not isinstance(op.result, str)
        or len(op.operands) != 2
        or op.result not in env
    ):
        return None
    result_shape = env[op.result]
    rank = len(result_shape)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if rank == 0 or rank > len(alphabet):
        return None
    output_labels = alphabet[:rank]
    operand_labels: list[str] = []
    saw_nontrailing_insertion = False
    for operand in op.operands:
        source_shape = env.get(operand)
        if source_shape is None or not source_shape:
            return None
        operand_axes = tuple(axes.get(operand, ()))
        if not operand_axes:
            if source_shape != result_shape:
                return None
            positions = tuple(range(rank))
        else:
            projected, inserted = _broadcast_projection(
                source_shape, operand_axes, result_shape
            )
            if projected is None:
                return None
            positions = tuple(
                index for index in range(rank) if index not in set(inserted)
            )
            if len(positions) != len(source_shape):
                return None
            if any(
                source_extent != result_shape[position]
                for source_extent, position in zip(source_shape, positions)
            ):
                return None
            leading = tuple(range(rank - len(source_shape)))
            if inserted and inserted != leading:
                saw_nontrailing_insertion = True
        operand_labels.append("".join(output_labels[index] for index in positions))
    if not saw_nontrailing_insertion:
        return None
    return f"{operand_labels[0]},{operand_labels[1]}->{output_labels}"


def _op_declared_shape(
    op: IntentOp,
    fold: int,
    extents: dict[str, int] | None,
) -> list[int] | None:
    if not isinstance(op.type, dict) or not isinstance(
        op.type.get("shape"), list
    ):
        return None
    return _resolve(op.type["shape"], extents or {})[fold:]


def _op_target_shape(
    op: IntentOp,
    env: Mapping[str, list[int]],
    fold: int,
    extents: dict[str, int] | None,
) -> list[int] | None:
    declared = _op_declared_shape(op, fold, extents)
    constrained = env.get(op.result) if isinstance(op.result, str) else None
    if (
        declared is not None
        and constrained is not None
        and declared != constrained
    ):
        raise ValueError(
            f"op result {op.result!r} declared shape {declared} conflicts "
            f"with consumer/output constraint {constrained}"
        )
    return declared if declared is not None else constrained


def _infer_op(
    op: IntentOp,
    env: dict[str, list[int]],
    fold: int,
    extents: dict[str, int] | None = None,
    broadcast_axes: BroadcastAxes | None = None,
) -> list[int] | None:
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
        axes = [int(a) - fold for a in op.attrs["axes"]]
        keepdims = list(shp)
        for ax in axes:
            keepdims[ax] = 1
        collapsed = [dim for index, dim in enumerate(shp) if index not in axes]
        # Canonical J carries a checked result type for every operation.  Use
        # that declaration to disambiguate collapsed versus keepdims reductions
        # even when the reduction is an intermediate value.  Looking only at
        # ``env`` loses the declaration until a downstream consumer happens to
        # constrain it and can silently choose the legacy keepdims default.
        declared = _op_target_shape(op, env, fold, extents)
        explicit_keepdims = op.attrs.get("keepdims")
        if explicit_keepdims is not None:
            if not isinstance(explicit_keepdims, bool):
                raise ValueError("reduce keepdims must be a boolean")
            selected = keepdims if explicit_keepdims else collapsed
            if declared is not None and declared != selected:
                raise ValueError(
                    f"reduce result {op.result!r} shape {declared} conflicts "
                    f"with keepdims={explicit_keepdims}: expected {selected}"
                )
            return list(selected)
        if declared is not None:
            if declared not in (keepdims, collapsed):
                raise ValueError(
                    f"reduce result {op.result!r} shape {declared} is neither "
                    f"keepdims {keepdims} nor collapsed {collapsed}"
                )
            return list(declared)
        if sorted(axes) == list(range(len(shp))):
            # A full reduction has no surviving semantic axis.  Rank-0 is the
            # canonical default and still broadcasts naturally to later ops;
            # keeping one unit dimension here would make scalar final-stage
            # reductions (sum(partials)/M) disagree with their ABI.
            return []
        for ax in axes:  # existing default: keepdims for broadcast consumers
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
        return [_resolve_int(op.attrs["n"], extents)]
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
    if op.op == "transpose":
        (x,) = op.operands
        if x not in env:
            return None
        perm = _fold_permutation(op.attrs["perm"], fold, extents or {})
        if sorted(perm) != list(range(len(env[x]))):
            raise ValueError(
                f"transpose perm {perm!r} is invalid for rank {len(env[x])}"
            )
        inferred = [env[x][axis] for axis in perm]
        declared = env.get(op.result)
        if declared is not None and declared != inferred:
            raise ValueError(
                f"transpose result {op.result!r} shape {declared} does not "
                f"match permuted input shape {inferred}"
            )
        return inferred
    if op.op == "map_unary":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_binary":
        a, b = op.operands
        axes = _op_broadcast_axes(op, broadcast_axes)
        combine = _broadcast_checked if axes else _broadcast
        declared = _op_target_shape(op, env, fold, extents)
        inferred = combine(
            _insert_broadcast_axes(
                env.get(a), axes.get(a, ()), declared
            ),
            _insert_broadcast_axes(
                env.get(b), axes.get(b, ()), declared
            ),
        )
        if declared is not None and axes and inferred != declared:
            raise ValueError(
                f"map_binary result {op.result!r} inferred shape {inferred} "
                f"conflicts with declared shape {declared}"
            )
        return inferred
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
        axes = _op_broadcast_axes(op, broadcast_axes)
        combine = _broadcast_checked if axes else _broadcast
        declared = _op_target_shape(op, env, fold, extents)
        inferred = combine(
            combine(
                _insert_broadcast_axes(
                    env.get(c), axes.get(c, ()), declared
                ),
                _insert_broadcast_axes(
                    env.get(x), axes.get(x, ()), declared
                ),
            ),
            _insert_broadcast_axes(
                env.get(y), axes.get(y, ()), declared
            ),
        )
        if declared is not None and axes and inferred != declared:
            raise ValueError(
                f"select result {op.result!r} inferred shape {inferred} "
                f"conflicts with declared shape {declared}"
            )
        return inferred
    if op.op == "value_if":
        if op.positive is None or op.negative is None:
            raise ValueError("value_if requires two explicit arms")
        if op.predicate in env and env[op.predicate] != []:
            raise ValueError("value_if predicate must be a rank-0 tensor")
        declared = _op_declared_shape(op, fold, extents)
        yields = [op.positive.yield_value, op.negative.yield_value]
        known = [env[name] for name in yields if name in env]
        if any(shape != known[0] for shape in known[1:]):
            raise ValueError("value_if arm yields have different shapes")
        if declared is not None and any(shape != declared for shape in known):
            raise ValueError(
                f"value_if result {op.result!r} shape {declared} differs "
                "from an arm yield"
            )
        return list(declared if declared is not None else known[0]) if (
            declared is not None or known
        ) else None
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
        raw_lo = [_resolve_int(v, extents) for v in op.attrs["lo"]]
        raw_hi = [_resolve_int(v, extents) for v in op.attrs["hi"]]
        expected_rank = len(env[x]) + fold
        if len(raw_lo) != expected_rank or len(raw_hi) != expected_rank:
            raise ValueError(
                f"pad lo/hi must have {expected_rank} entries, got "
                f"{len(raw_lo)}/{len(raw_hi)}"
            )
        if any(value != 0 for value in raw_lo[:fold] + raw_hi[:fold]):
            raise ValueError("emit_mlir: cannot fold padded batch dimensions")
        lo, hi = raw_lo[fold:], raw_hi[fold:]
        if any(value < 0 for value in lo + hi):
            raise ValueError("pad lo/hi entries must be non-negative")
        inferred = [d + lo[i] + hi[i] for i, d in enumerate(env[x])]
        declared = env.get(op.result)
        if declared is not None and declared != inferred:
            raise ValueError(
                f"pad result {op.result!r} shape {declared} does not match "
                f"input+lo+hi shape {inferred}"
            )
        return inferred
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
            "logical_and", "logical_or", "logical_xor", "and", "or", "xor",
            "isclose"}
# map_unary fns whose result is an i1 (finiteness predicates + i1 complement).
_U_TO_I1 = {"isnan", "isinf", "isfinite", "logical_not"}

_FLOAT_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})
_INTEGER_DTYPES = frozenset({"i1", "i8", "i16", "i32", "i64"})
_DTYPES = _FLOAT_DTYPES | _INTEGER_DTYPES


def _normalize_dtype(value: object, context: str) -> str:
    """Return one supported MLIR scalar dtype, normalizing JSON ``bool``."""

    dtype = "i1" if value == "bool" else value
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError(f"emit_mlir: invalid dtype {value!r} for {context}")
    return dtype


def _value_dimension_attribute(
    fn: IntentFunction,
    equalities: Sequence[tuple[int, int]],
) -> str | None:
    """Render the closed J-derived rank-0 scalar equality contract.

    The pair index is an ``IntentFunction.inputs`` index, which is exactly the
    function-argument order emitted below.  This helper accepts no J, C, pi, or
    Harness object and cannot invent an equality.
    """

    checked: list[tuple[int, int]] = []
    seen: set[int] = set()
    for ordinal, equality in enumerate(equalities):
        if (
            not isinstance(equality, (tuple, list))
            or len(equality) != 2
        ):
            raise ValueError(
                "emit_mlir: value/dimension equality "
                f"{ordinal} must be an (input index, value) pair"
            )
        argument_index, value = equality
        if (
            isinstance(argument_index, bool)
            or not isinstance(argument_index, int)
            or argument_index < 0
            or argument_index >= len(fn.inputs)
        ):
            raise ValueError(
                "emit_mlir: value/dimension equality references an invalid "
                "IntentFunction input index"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "emit_mlir: value/dimension equality value must be an integer"
            )
        if argument_index in seen:
            raise ValueError(
                "emit_mlir: duplicate value/dimension equality input index"
            )
        tensor = fn.inputs[argument_index]
        dtype = _normalize_dtype(
            tensor.dtype,
            f"value/dimension input {tensor.name!r}",
        )
        if tensor.shape or dtype not in _INTEGER_DTYPES or dtype == "i1":
            raise ValueError(
                "emit_mlir: value/dimension equality requires a rank-0 "
                "non-boolean integer input"
            )
        width = int(dtype[1:])
        minimum = -(1 << (width - 1))
        maximum = (1 << (width - 1)) - 1
        if value < minimum or value > maximum:
            raise ValueError(
                "emit_mlir: value/dimension equality is not representable by "
                f"input dtype {dtype}"
            )
        seen.add(argument_index)
        checked.append((argument_index, value))
    if not checked:
        return None
    flattened = ", ".join(
        str(item)
        for equality in sorted(checked)
        for item in equality
    )
    return "intent.scalar_equalities = array<i64: " + flattened + ">"


def _function_attributes(
    fn: IntentFunction,
    *,
    value_dimension_equalities: Sequence[tuple[int, int]],
    inplace_outputs: Sequence[int],
    paired_rotation: tuple[str, tuple[int, int, int, int], tuple[int, ...]]
    | None,
) -> str:
    attributes: list[str] = []
    equality = _value_dimension_attribute(fn, value_dimension_equalities)
    if equality is not None:
        attributes.append(equality)

    if inplace_outputs:
        if len(inplace_outputs) != len(fn.outputs):
            raise ValueError(
                "emit_mlir: inplace output mapping must cover every result"
            )
        checked: list[int] = []
        for output_index, argument_index in enumerate(inplace_outputs):
            if (
                isinstance(argument_index, bool)
                or not isinstance(argument_index, int)
                or argument_index < -1
                or argument_index >= len(fn.inputs)
            ):
                raise ValueError("emit_mlir: invalid inplace input index")
            if argument_index >= 0:
                source = fn.inputs[argument_index]
                output = fn.outputs[output_index]
                if source.dtype != output.dtype or source.shape != output.shape:
                    raise ValueError(
                        "emit_mlir: inplace source/result tensor types differ"
                    )
            checked.append(argument_index)
        flattened = ", ".join(str(item) for item in checked)
        attributes.append(f"intent.inplace = array<i64: {flattened}>")

    if paired_rotation is not None:
        layout, input_indices, sections = paired_rotation
        if layout not in {"shared_position", "three_plane", "interleaved"}:
            raise ValueError("emit_mlir: unknown paired rotation layout")
        if (
            len(input_indices) != 4
            or len(set(input_indices)) != 4
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(fn.inputs)
                for index in input_indices
            )
        ):
            raise ValueError("emit_mlir: invalid paired rotation input mapping")
        if layout == "three_plane":
            if len(sections) != 3 or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in sections
            ):
                raise ValueError("emit_mlir: invalid three-plane sections")
        elif sections:
            raise ValueError(
                "emit_mlir: non-three-plane rotation has no sections"
            )
        arguments = ", ".join(str(item) for item in input_indices)
        attributes.extend(
            (
                'intent.paired_rotation_version = '
                '"intentir.paired-last-axis-rotation.v1"',
                f'intent.paired_rotation = "{layout}"',
                f"intent.paired_rotation_inputs = array<i64: {arguments}>",
            )
        )
        if sections:
            values = ", ".join(str(item) for item in sections)
            attributes.append(
                f"intent.paired_rotation_sections = array<i64: {values}>"
            )

    return "" if not attributes else " attributes {" + ", ".join(attributes) + "}"


def _promote_dtype(lhs: str | None, rhs: str | None) -> str | None:
    """Small deterministic promotion lattice used by mixed-operand ops.

    Same-typed operands stay in that type.  Mixed low-precision floats promote
    to f32, as do float/integer combinations involving f32; f64 dominates all
    other types.  This mirrors the paper validator's intended inference without
    reinstating a blanket f32 execution domain.
    """

    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs == rhs:
        return lhs
    if lhs in _FLOAT_DTYPES or rhs in _FLOAT_DTYPES:
        floats = {dtype for dtype in (lhs, rhs) if dtype in _FLOAT_DTYPES}
        if "f64" in floats:
            return "f64"
        if "f32" in floats or floats == {"f16", "bf16"}:
            return "f32"
        return next(iter(floats))
    widths = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}
    if lhs in widths and rhs in widths:
        return lhs if widths[lhs] >= widths[rhs] else rhs
    return None


def _merge_operand_dtypes(
    op: IntentOp, dtypes: dict[str, str], start: int = 0
) -> str | None:
    result: str | None = None
    for name in op.operands[start:]:
        result = _promote_dtype(result, dtypes.get(name))
    return result


def _declared_op_dtype(op: IntentOp) -> str | None:
    if not isinstance(op.type, dict) or "dtype" not in op.type:
        return None
    return _normalize_dtype(op.type["dtype"], f"{op.op} result type")


def _cast_target_dtype(attrs: Mapping[str, object]) -> str:
    """Read canonical `to_dtype` and the archived `dtype` spelling."""

    value = attrs.get("to_dtype", attrs.get("dtype"))
    return _normalize_dtype(value, "cast target dtype")


def _result_names(op: IntentOp) -> list[str]:
    if isinstance(op.result, list):
        return list(op.result)
    return [op.result] if isinstance(op.result, str) else []


def _infer_op_dtypes(op: IntentOp, dtypes: dict[str, str]) -> dict[str, str]:
    """Infer the element type(s) produced by one atomic Intent operation."""

    names = _result_names(op)
    if not names:
        return {}
    a = op.attrs
    inferred: list[str | None]

    if op.op == "cast":
        inferred = [_cast_target_dtype(a)]
    elif op.op == "contraction":
        # Accumulator/result precision is value semantics, not Layer B.  An
        # explicit op.type wins; an externally declared result type is the next
        # source of truth, otherwise operands determine the type.
        dtype = _declared_op_dtype(op) or dtypes.get(names[0])
        inferred = [dtype or _merge_operand_dtypes(op, dtypes)]
    elif op.op == "reduce":
        inferred = [
            "i1" if a.get("fn") in ("all", "any")
            else dtypes.get(op.operands[0])
        ]
    elif op.op == "map_unary":
        inferred = [
            "i1" if a.get("fn") in _U_TO_I1 else dtypes.get(op.operands[0])
        ]
    elif op.op == "map_binary":
        inferred = [
            "i1" if a.get("fn") in _CMP_FNS
            else _merge_operand_dtypes(op, dtypes)
        ]
    elif op.op == "select":
        inferred = [_merge_operand_dtypes(op, dtypes, start=1)]
    elif op.op == "value_if":
        if op.positive is None or op.negative is None:
            raise ValueError("value_if requires two explicit arms")
        arm_dtypes = [
            dtypes.get(op.positive.yield_value),
            dtypes.get(op.negative.yield_value),
        ]
        known = [dtype for dtype in arm_dtypes if dtype is not None]
        if len(set(known)) > 1:
            raise ValueError("value_if arm yields have different dtypes")
        inferred = [_declared_op_dtype(op) or (known[0] if known else None)]
    elif op.op in {
        "scan", "transpose", "gather", "scatter", "masked_select",
        "masked_scatter", "concat", "pad", "tile", "flip", "triu",
        "pool", "upsample",
    }:
        inferred = [dtypes.get(op.operands[0])]
        if op.op == "pool" and len(names) == 2:
            inferred.append("i64")
    elif op.op == "sort":
        source = dtypes.get(op.operands[0])
        if len(names) == 2:
            inferred = [source, "i32"]
        elif a.get("return") == "indices":
            inferred = ["i32"]
        else:
            inferred = [source]
    elif op.op == "arg_reduce":
        inferred = ["i32"]
    elif op.op == "iota":
        inferred = [_normalize_dtype(a.get("dtype", "i32"), "iota dtype")]
    elif op.op == "nonzero":
        inferred = ["i64"]
    elif op.op in ("const", "rand"):
        inferred = [_normalize_dtype(a.get("dtype", "f32"), f"{op.op} dtype")]
    elif op.op in ("conv", "polar"):
        common = _merge_operand_dtypes(op, dtypes)
        inferred = [common] * len(names)
    else:
        inferred = [None] * len(names)

    declared = _declared_op_dtype(op)
    if declared is not None and len(names) == 1:
        if inferred[0] is not None and inferred[0] != declared:
            raise ValueError(
                f"emit_mlir: {op.op} result {names[0]!r} declares {declared} "
                f"but its semantics infer {inferred[0]}"
            )
        inferred[0] = declared
    return {
        name: dtype for name, dtype in zip(names, inferred) if dtype is not None
    }


def _record_dtype(
    dtypes: dict[str, str], name: str, dtype: str, context: str
) -> bool:
    previous = dtypes.get(name)
    if previous is not None and previous != dtype:
        raise ValueError(
            f"emit_mlir: dtype conflict for {name!r}: {previous} versus "
            f"{dtype} ({context})"
        )
    if previous is None:
        dtypes[name] = dtype
        return True
    return False


def _infer_body_dtypes(
    body: list[IntentOp], dtypes: dict[str, str]
) -> bool:
    changed = False
    for op in body:
        if op.op == "loop":
            # A semantic loop is a region operation.  Its scan block arguments
            # inherit the source element type; carry block arguments and results
            # inherit the authored state type (or, for older artifacts, the
            # initializer/yield type).  Seed those region values before walking
            # the body so nested loops use the same inference as top-level ones.
            for scan in op.scan:
                source_dtype = dtypes.get(scan.source)
                if source_dtype is not None:
                    changed |= _record_dtype(
                        dtypes, scan.name, source_dtype, "loop scan slice"
                    )
            for state in op.carry:
                declared = (
                    _normalize_dtype(
                        state.type.dtype, "loop carry result type"
                    )
                    if state.type is not None
                    else None
                )
                state_dtype = declared or dtypes.get(state.init_value)
                if state_dtype is not None:
                    changed |= _record_dtype(
                        dtypes,
                        state.body_argument,
                        state_dtype,
                        "loop carry block argument",
                    )
                    changed |= _record_dtype(
                        dtypes,
                        state.result_value,
                        state_dtype,
                        "loop carry result",
                    )
            changed |= _infer_body_dtypes(op.body, dtypes)
            for state in op.carry:
                yielded_dtype = dtypes.get(state.next_value)
                if yielded_dtype is None:
                    continue
                changed |= _record_dtype(
                    dtypes,
                    state.body_argument,
                    yielded_dtype,
                    "loop yielded carry",
                )
                changed |= _record_dtype(
                    dtypes,
                    state.result_value,
                    yielded_dtype,
                    "loop result",
                )
            continue
        for arm in (op.positive, op.negative):
            if arm is not None:
                changed |= _infer_body_dtypes(arm.body, dtypes)
        for name, dtype in _infer_op_dtypes(op, dtypes).items():
            changed |= _record_dtype(dtypes, name, dtype, op.op)
    return changed


def _flat_dtypes(fn: IntentFunction) -> dict[str, str]:
    dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"input {tensor.name!r}")
        for tensor in fn.inputs
    }
    for tensor in fn.outputs:
        _record_dtype(
            dtypes,
            tensor.name,
            _normalize_dtype(tensor.dtype, f"output {tensor.name!r}"),
            "function output ABI",
        )
    for _ in range(len(fn.body) + 3):
        if not _infer_body_dtypes(fn.body, dtypes):
            break
    for tensor in fn.outputs:
        declared = _normalize_dtype(tensor.dtype, f"output {tensor.name!r}")
        actual = dtypes.get(tensor.name)
        if actual is None:
            raise ValueError(
                f"emit_mlir: cannot infer dtype for output {tensor.name!r}"
            )
        if actual != declared:
            raise ValueError(
                f"emit_mlir: output {tensor.name!r} declares {declared} but "
                f"the produced SSA value has dtype {actual}; add an explicit cast"
            )
    return dtypes


def _loop_dtypes(fn: IntentFunction, loop: IntentOp) -> dict[str, str]:
    dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"input {tensor.name!r}")
        for tensor in fn.inputs
    }
    for scan in loop.scan:
        source = scan["from"]
        _record_dtype(dtypes, scan["name"], dtypes[source], "loop scan slice")

    result_names = _result_names(loop)
    carries = [item["name"] for item in loop.carry]
    for item in loop.carry:
        init = item.get("init")
        if isinstance(init, str) and init in dtypes:
            _record_dtype(dtypes, item["name"], dtypes[init], "loop carry init")
    output_dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"output {tensor.name!r}")
        for tensor in fn.outputs
    }
    for result, carry in zip(result_names, carries):
        if result in output_dtypes:
            _record_dtype(dtypes, carry, output_dtypes[result], "loop result ABI")

    for _ in range(len(loop.body) + len(carries) + 3):
        changed = _infer_body_dtypes(loop.body, dtypes)
        for carry, yielded, result in zip(carries, loop.yields, result_names):
            yielded_dtype = dtypes.get(yielded)
            if yielded_dtype is not None:
                changed |= _record_dtype(
                    dtypes, carry, yielded_dtype, "loop yielded carry"
                )
                changed |= _record_dtype(
                    dtypes, result, yielded_dtype, "loop result"
                )
        if not changed:
            break

    required = set(carries) | set(loop.yields)
    for op in loop.body:
        required.update(op.operands)
        required.update(_result_names(op))
    missing = sorted(name for name in required if name not in dtypes)
    if missing:
        raise ValueError(
            "emit_mlir: cannot infer loop-body dtype(s): " + ", ".join(missing)
        )
    return dtypes


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


def _mlir_dense_float(v, dtype: str) -> str:
    """Format one dense floating literal, including IEEE non-finite values."""

    value = float(v)
    if math.isfinite(value):
        return _mlir_f(value)
    normalized = dtype.removeprefix("torch.").lower()
    encodings = {
        "f16": ("0x7C00", "0xFC00", "0x7E00"),
        "float16": ("0x7C00", "0xFC00", "0x7E00"),
        "bf16": ("0x7F80", "0xFF80", "0x7FC0"),
        "bfloat16": ("0x7F80", "0xFF80", "0x7FC0"),
        "f32": ("0x7F800000", "0xFF800000", "0x7FC00000"),
        "float32": ("0x7F800000", "0xFF800000", "0x7FC00000"),
        "f64": (
            "0x7FF0000000000000",
            "0xFFF0000000000000",
            "0x7FF8000000000000",
        ),
        "float64": (
            "0x7FF0000000000000",
            "0xFFF0000000000000",
            "0x7FF8000000000000",
        ),
    }
    if normalized not in encodings:
        raise ValueError(
            f"emit_mlir: non-finite dense literal has unsupported dtype {dtype!r}"
        )
    positive_inf, negative_inf, quiet_nan = encodings[normalized]
    if math.isnan(value):
        return quiet_nan
    return negative_inf if value < 0 else positive_inf


def _infer_body(
    body: list[IntentOp],
    env: dict[str, list[int]],
    fold: int,
    extents: dict[str, int] | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    loop_schedule_tile: int | None = None,
) -> None:
    for _ in range(len(body) + 2):
        changed = False
        # Structural consumers can constrain an otherwise ambiguous producer.
        # Most importantly, a declared output after transpose determines the
        # transpose operand rank/shape, which lets an intermediate reduce choose
        # collapsed vs keepdims without relying on a model-specific heuristic.
        for op in reversed(body):
            if not isinstance(op.result, str) or op.result not in env:
                continue
            result_shape = env[op.result]
            if op.op in {"map_unary", "cast"} and len(op.operands) == 1:
                operand = op.operands[0]
                if operand not in env:
                    env[operand] = list(result_shape)
                    changed = True
                continue
            if op.op in {"map_binary", "select"}:
                axes = _op_broadcast_axes(op, broadcast_axes)
                if not axes:
                    continue
                for operand in op.operands:
                    if operand not in env and not axes.get(operand):
                        env[operand] = list(result_shape)
                        changed = True
                continue
            if (
                op.op != "transpose"
                or len(op.operands) != 1
            ):
                continue
            operand = op.operands[0]
            perm = _fold_permutation(op.attrs["perm"], fold, extents or {})
            if sorted(perm) != list(range(len(result_shape))):
                raise ValueError(
                    f"transpose perm {perm!r} is invalid for declared result "
                    f"rank {len(result_shape)}"
                )
            inferred_operand = [0] * len(result_shape)
            for out_dim, input_dim in enumerate(perm):
                inferred_operand[input_dim] = result_shape[out_dim]
            previous = env.get(operand)
            if previous is None:
                env[operand] = inferred_operand
                changed = True
            elif previous != inferred_operand:
                raise ValueError(
                    f"transpose operand {operand!r} shape {previous} conflicts "
                    f"with result constraint {inferred_operand}"
                )
        for op in body:
            if op.op == "loop":
                if op.trip is None:
                    raise ValueError("emit_mlir: loop requires a semantic trip axis")
                tile = _tile(op, extents or {}, loop_schedule_tile)
                for scan in op.scan:
                    source_shape = env.get(scan.source)
                    if source_shape is None:
                        continue
                    axis = int(scan.axis) - fold
                    if axis < 0 or axis >= len(source_shape):
                        raise ValueError(
                            "emit_mlir: loop scan axis is outside its source rank"
                        )
                    slice_shape = list(source_shape)
                    slice_shape[axis] = tile
                    if env.get(scan.name) != slice_shape:
                        env[scan.name] = slice_shape
                        changed = True
                for state in op.carry:
                    declared = None
                    if state.type is not None:
                        declared = _resolve(
                            state.type.shape, extents or {}
                        )[fold:]
                    init_shape = env.get(state.init_value)
                    state_shape = declared or init_shape
                    if state_shape is not None:
                        if init_shape is not None and declared is not None and (
                            init_shape != declared
                        ):
                            raise ValueError(
                                "emit_mlir: loop carry initializer shape differs "
                                "from its authored state type"
                            )
                        if env.get(state.body_argument) != state_shape:
                            env[state.body_argument] = list(state_shape)
                            changed = True
                before = dict(env)
                body_extents = dict(extents or {})
                body_extents[op.trip.axis] = tile
                _infer_body(
                    op.body,
                    env,
                    fold,
                    body_extents,
                    broadcast_axes,
                    loop_schedule_tile,
                )
                changed |= env != before
                for state in op.carry:
                    state_shape = env.get(state.body_argument)
                    yielded_shape = env.get(state.next_value)
                    if (
                        state_shape is not None
                        and yielded_shape is not None
                        and state_shape != yielded_shape
                    ):
                        raise ValueError(
                            "emit_mlir: loop yielded value shape differs from carry"
                        )
                    result_shape = yielded_shape or state_shape
                    if result_shape is not None and (
                        env.get(state.result_value) != result_shape
                    ):
                        env[state.result_value] = list(result_shape)
                        changed = True
                continue
            for arm in (op.positive, op.negative):
                if arm is not None:
                    before = dict(env)
                    _infer_body(
                        arm.body,
                        env,
                        fold,
                        extents,
                        broadcast_axes,
                        loop_schedule_tile,
                    )
                    changed |= env != before
            sh = _infer_op(op, env, fold, extents, broadcast_axes)
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
    def __init__(
        self,
        dtypes: dict[str, str],
        broadcast_axes: BroadcastAxes | None = None,
        proven_index_bounds: IndexBoundsProjection | None = None,
        loop_schedule_tile: int | None = None,
    ) -> None:
        self.lines: list[str] = []
        self._n = 0
        self.ssa: dict[str, str] = {}   # JSON name -> SSA name
        self.dtypes = dtypes
        self.broadcast_axes = broadcast_axes or {}
        self.proven_index_bounds = proven_index_bounds or frozenset()
        self.loop_schedule_tile = loop_schedule_tile

    def _ttn(self, name: str, env: dict[str, list[int]]) -> str:
        """Tensor type for one JSON value from the inferred dtype environment."""
        if name not in self.dtypes:
            raise ValueError(f"emit_mlir: no dtype inferred for {name!r}")
        return _tt(env[name], self.dtypes[name])

    def _ttn_shape(self, name: str, shape: Sequence[int]) -> str:
        if name not in self.dtypes:
            raise ValueError(f"emit_mlir: no dtype inferred for {name!r}")
        return _tt(list(shape), self.dtypes[name])

    def _broadcast_operand(
        self,
        name: str,
        env: dict[str, list[int]],
        axes: Sequence[int],
        indent: str,
        consumer_shape: Sequence[int] | None = None,
        element_dtype: str | None = None,
    ) -> tuple[str, str]:
        source_shape = env[name]
        target_shape, inserted_axes = _broadcast_projection(
            source_shape, axes, consumer_shape
        )
        assert target_shape is not None
        source_type = self._ttn(name, env)
        value = self.ssa[name]
        projected_dtype = element_dtype or self.dtypes[name]
        projected_source_type = _tt(source_shape, projected_dtype)
        if projected_dtype != self.dtypes[name]:
            self._n += 1
            promoted = f"%{name}_promote_{self._n}"
            self.lines.append(
                f'{indent}{promoted} = intent.cast {value} '
                f'{{to_dtype = "{projected_dtype}"}} : '
                f'({source_type}) -> {projected_source_type}'
            )
            value = promoted
        target_type = _tt(target_shape, projected_dtype)
        # Rank-0 tensors already broadcast through an empty affine input map;
        # no tensor.expand_shape is needed for explicit scalar axes.
        if not axes or not source_shape or target_shape == source_shape:
            return value, projected_source_type
        rank_gap = len(target_shape) - len(source_shape)
        standard_trailing_vector = (
            consumer_shape is not None
            and len(source_shape) == 1
            and inserted_axes == tuple(range(rank_gap))
            and source_shape == target_shape[rank_gap:]
        )
        if standard_trailing_vector:
            # Intent map lowering and the row fast path both understand the
            # conventional trailing rank-1 parameter broadcast directly.  Keep
            # it rank-1 so specialized row kernels can classify it as a column
            # parameter; non-trailing projections still materialize an explicit
            # expand_shape below.
            return value, projected_source_type

        inserted = set(inserted_axes)
        source_positions = [
            index for index in range(len(target_shape)) if index not in inserted
        ]
        reassociation: list[list[int]] = []
        for index, position in enumerate(source_positions):
            start = 0 if index == 0 else position
            end = (
                source_positions[index + 1]
                if index + 1 < len(source_positions)
                else len(target_shape)
            )
            reassociation.append(list(range(start, end)))
        groups = "[" + ", ".join(
            "[" + ", ".join(str(value) for value in group) + "]"
            for group in reassociation
        ) + "]"
        output_shape = ", ".join(str(value) for value in target_shape)
        self._n += 1
        result = f"%{name}_broadcast_{self._n}"
        self.lines.append(
            f"{indent}{result} = tensor.expand_shape {value} {groups} "
            f"output_shape [{output_shape}] : {projected_source_type} "
            f"into {target_type}"
        )
        return result, target_type

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
                indent: str = "    ",
                extents: dict[str, int] | None = None,
                node_ref: str | None = None) -> None:
        a = op.attrs
        if op.op == "loop":
            if op.trip is None or not op.carry or not op.scan:
                raise ValueError(
                    "emit_mlir: loop requires trip, scan inputs, and carry state"
                )
            if len(op.yields) != len(op.carry):
                raise ValueError("emit_mlir: loop carry/yield arity mismatch")
            resolved_extents = dict(extents or {})
            tile = _tile(op, resolved_extents, self.loop_schedule_tile)
            body_extents = dict(resolved_extents)
            body_extents[op.trip.axis] = tile
            scan_values: list[str] = []
            scan_types: list[str] = []
            scan_axes: list[str] = []
            for scan in op.scan:
                if scan.source not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop scan source {scan.source!r} is unavailable"
                    )
                scan_values.append(self.ssa[scan.source])
                scan_types.append(self._ttn(scan.source, env))
                scan_axes.append(f"{int(scan.axis) - fold} : i64")

            init_values: list[str] = []
            init_types: list[str] = []
            result_types: list[str] = []
            for state in op.carry:
                if state.init_value not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop carry init {state.init_value!r} is unavailable"
                    )
                init_values.append(self.ssa[state.init_value])
                init_types.append(self._ttn(state.init_value, env))
                result_types.append(self._ttn(state.result_value, env))

            self._n += 1
            loop_value = f"%intentir_loop_{self._n}"
            self.lines.append(
                f"{indent}{loop_value}:{len(op.carry)} = intent.loop "
                f"scan({', '.join(scan_values)} : {', '.join(scan_types)}) "
                f"init({', '.join(init_values)} : {', '.join(init_types)}) "
                f"{{scan_axes = [{', '.join(scan_axes)}], "
                f"tile_hint = {tile} : i64}} -> {', '.join(result_types)} do {{"
            )

            outer_ssa = dict(self.ssa)
            block_args: list[str] = []
            for scan in op.scan:
                block = self.fresh(scan.name)
                block_args.append(f"{block}: {self._ttn(scan.name, env)}")
            for state in op.carry:
                block = self.fresh(state.body_argument)
                block_args.append(
                    f"{block}: {self._ttn(state.body_argument, env)}"
                )
            self.lines.append(f"{indent}^bb0({', '.join(block_args)}):")
            for child_index, child in enumerate(op.body):
                self.emit_op(
                    child,
                    env,
                    fold,
                    indent=indent + "  ",
                    extents=body_extents,
                    node_ref=(
                        f"{node_ref}.body[{child_index}]"
                        if node_ref is not None
                        else None
                    ),
                )
            yielded_values = []
            yielded_types = []
            for yielded in op.yields:
                if yielded not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop yield {yielded!r} is unavailable"
                    )
                yielded_values.append(self.ssa[yielded])
                yielded_types.append(self._ttn(yielded, env))
            self.lines.append(
                f"{indent}  intent.yield {', '.join(yielded_values)} : "
                f"{', '.join(yielded_types)}"
            )
            self.lines.append(f"{indent}}}")
            self.ssa = outer_ssa
            for index, state in enumerate(op.carry):
                self.ssa[state.result_value] = f"{loop_value}#{index}"
            return
        if op.op == "value_if":
            if (
                not isinstance(op.result, str)
                or op.positive is None
                or op.negative is None
                or not isinstance(op.predicate, str)
            ):
                raise ValueError(
                    "emit_mlir: value_if requires one result, predicate, and two arms"
                )
            if env.get(op.predicate) != []:
                raise ValueError(
                    "emit_mlir: value_if predicate must be a rank-0 tensor"
                )
            if self.dtypes.get(op.predicate) != "i1":
                raise ValueError("emit_mlir: value_if predicate must have i1 dtype")

            condition_tensor = self.ssa[op.predicate]
            self._n += 1
            condition = f"%{op.result}_condition_{self._n}"
            self.lines.append(
                f"{indent}{condition} = tensor.extract {condition_tensor}[] : "
                f"{self._ttn(op.predicate, env)}"
            )
            self._n += 1
            result = f"%{op.result}_{self._n}"
            result_type = self._ttn(op.result, env)
            self.lines.append(
                f"{indent}{result} = scf.if {condition} -> ({result_type}) {{"
            )

            outer_ssa = dict(self.ssa)

            def emit_arm(arm, arm_name: str) -> None:
                self.ssa = dict(outer_ssa)
                for child_index, child in enumerate(arm.body):
                    self.emit_op(
                        child,
                        env,
                        fold,
                        indent=indent + "  ",
                        extents=extents,
                        node_ref=(
                            f"{node_ref}.{arm_name}[{child_index}]"
                            if node_ref is not None
                            else None
                        ),
                    )
                yielded = arm.yield_value
                if yielded not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield "
                        f"{yielded!r} is unavailable"
                    )
                if env.get(yielded) != env[op.result]:
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield shape "
                        "differs from its result"
                    )
                if self.dtypes.get(yielded) != self.dtypes.get(op.result):
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield dtype "
                        "differs from its result"
                    )
                self.lines.append(
                    f"{indent}  scf.yield {self.ssa[yielded]} : {result_type}"
                )

            emit_arm(op.positive, "positive")
            self.lines.append(f"{indent}}} else {{")
            emit_arm(op.negative, "negative")
            self.lines.append(f"{indent}}}")
            self.ssa = outer_ssa
            self.ssa[op.result] = result
            return
        if op.op == "cast":
            to = _cast_target_dtype(a)
            operand = op.operands[0]
            if self.dtypes[op.result] != to:
                raise ValueError(
                    f"emit_mlir: cast result {op.result!r} inferred as "
                    f"{self.dtypes[op.result]}, expected {to}"
                )
            rr = self.fresh(op.result)
            x = self.ssa[operand]
            self.lines.append(
                f'{indent}{rr} = intent.cast {x} {{to_dtype = "{to}"}} : '
                f'({self._ttn(operand, env)}) -> '
                f'{self._ttn(op.result, env)}')
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
            x = self.ssa[op.operands[0]]
            desc = "true" if a.get("descending", False) else "false"
            value_dtype = self.dtypes[op.operands[0]]
            vty = _tt(shp, value_dtype)
            ity = "tensor<" + "x".join(str(s) for s in shp) + "xi32>"
            axis = int(a.get("axis", len(env[op.operands[0]]) - 1 + fold)) - fold
            self.lines.append(
                f'{indent}{vssa}, {issa} = intent.sort {x} '
                f'{{descending = {desc}, axis = {axis} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> ({vty}, {ity})')
            return
        if op.op == "pool":
            # avg -> one dtype-preserving output; max may carry a 2nd i64 index
            # (max_pool2d_with_indices). op.result is a list [out, indices] when
            # indices are named, else a single str.
            x = self.ssa[op.operands[0]]
            if isinstance(op.result, list):
                oname, iname = op.result[0], op.result[1]
            else:
                oname, iname = op.result, None
            oshp = env[oname]
            ossa = self.fresh(oname)
            oty = self._ttn(oname, env)
            attrs = (f'fn = "{a["fn"]}", kH = {int(a["kH"])} : i64, '
                     f'kW = {int(a["kW"])} : i64, sH = {int(a["sH"])} : i64, '
                     f'sW = {int(a["sW"])} : i64, padH = {int(a["padH"])} : i64, '
                     f'padW = {int(a["padW"])} : i64')
            if iname is not None:
                issa = self.fresh(iname)
                ity = "tensor<" + "x".join(str(s) for s in oshp) + "xi64>"
                self.lines.append(
                    f'{indent}{ossa}, {issa} = intent.pool {x} {{{attrs}}} : '
                    f'({self._ttn(op.operands[0], env)}) -> ({oty}, {ity})')
            else:
                self.lines.append(
                    f'{indent}{ossa} = intent.pool {x} {{{attrs}}} : '
                    f'({self._ttn(op.operands[0], env)}) -> {oty}')
            return
        if op.op == "polar":
            # abs,angle -> (real, imag) real/imag pair in the common float dtype.
            absv, ang = (self.ssa[o] for o in op.operands)
            rname, iname = op.result[0], op.result[1]
            rssa, issa = self.fresh(rname), self.fresh(iname)
            ta = self._ttn(op.operands[0], env)
            tb = self._ttn(op.operands[1], env)
            self.lines.append(
                f'{indent}{rssa}, {issa} = intent.polar {absv}, {ang} : '
                f'({ta}, {tb}) -> '
                f'({self._ttn(rname, env)}, {self._ttn(iname, env)})')
            return
        r = self.fresh(op.result)
        if op.op == "conv":
            # input, weight, optional bias -> single output. stride/padding/
            # dilation are length-R spatial arrays; groups partitions channels.
            x = self.ssa[op.operands[0]]
            w = self.ssa[op.operands[1]]
            tx, tw = (self._ttn(name, env) for name in op.operands[:2])
            st = ", ".join(str(int(v)) for v in a["stride"])
            pd = ", ".join(str(int(v)) for v in a["padding"])
            dl = ", ".join(str(int(v)) for v in a["dilation"])
            grp = int(a.get("groups", 1))
            attrs = (f'stride = [{st}], padding = [{pd}], dilation = [{dl}], '
                     f'groups = {grp} : i64')
            if len(op.operands) > 2:
                bn = op.operands[2]
                bssa, tb = self.ssa[bn], self._ttn(bn, env)
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w}, {bssa} {{{attrs}}} : '
                    f'({tx}, {tw}, {tb}) -> {self._ttn(op.result, env)}')
            else:
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w} {{{attrs}}} : '
                    f'({tx}, {tw}) -> {self._ttn(op.result, env)}')
        elif op.op == "upsample":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.upsample {x} {{mode = "nearest"}} : '
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "contraction":
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = (self._ttn(name, env) for name in op.operands)
            ins, out = a["subscripts"].split("->")
            aL, bL = (s.strip() for s in ins.split(","))
            sub = f"{_strip(aL, fold)},{_strip(bL, fold)}->{_strip(out.strip(), fold)}"
            self.lines.append(
                f'{indent}{r} = intent.contraction {x}, {y} '
                f'{{subscripts = "{sub}"}} : ({tx}, {ty}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "reduce":
            # all/any become i1; numeric reductions preserve the operand dtype.
            # Both decisions were made before printing by _infer_op_dtypes.
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
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "arg_reduce":
            # Index-of-extremum: an INTEGER (i32) result with `axis` collapsed.
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.arg_reduce {x} {{fn = "{a["fn"]}", '
                f'axis = {int(a["axis"]) - fold} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "map_unary":
            # Predicate fns (isnan/isinf/isfinite/logical_not) yield an i1; every
            # other fn is element-type-preserving.
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.map_unary {x} {{fn = "{a["fn"]}"}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "map_binary":
            axes = _op_broadcast_axes(op, self.broadcast_axes)
            contraction = _broadcast_mul_contraction_subscripts(op, env, axes)
            operand_dtype = _promote_dtype(
                self.dtypes.get(op.operands[0]),
                self.dtypes.get(op.operands[1]),
            )
            if operand_dtype is None:
                raise ValueError(
                    f"emit_mlir: map_binary {op.result!r} has no closed dtype "
                    "promotion"
                )
            if self.dtypes[op.operands[0]] != self.dtypes[op.operands[1]]:
                # The outer-product contraction spelling has its own mixed-
                # precision accumulator rules.  A semantic map promotion is
                # instead made explicit on each operand below.
                contraction = None
            if contraction is not None:
                x, y = (self.ssa[name] for name in op.operands)
                tx, ty = (self._ttn(name, env) for name in op.operands)
                self.lines.append(
                    f'{indent}{r} = intent.contraction {x}, {y} '
                    f'{{subscripts = "{contraction}"}} : ({tx}, {ty}) -> '
                    f'{self._ttn(op.result, env)}'
                )
            else:
                x, tx = self._broadcast_operand(
                    op.operands[0],
                    env,
                    axes.get(op.operands[0], ()),
                    indent,
                    env[op.result],
                    element_dtype=operand_dtype,
                )
                y, ty = self._broadcast_operand(
                    op.operands[1],
                    env,
                    axes.get(op.operands[1], ()),
                    indent,
                    env[op.result],
                    element_dtype=operand_dtype,
                )
                self.lines.append(
                    f'{indent}{r} = intent.map_binary {x}, {y} '
                    f'{{fn = "{a["fn"]}"}} : ({tx}, {ty}) -> '
                    f'{self._ttn(op.result, env)}'
                )
        elif op.op == "select":
            axes = _op_broadcast_axes(op, self.broadcast_axes)
            c, tc = self._broadcast_operand(
                op.operands[0],
                env,
                axes.get(op.operands[0], ()),
                indent,
                env[op.result],
            )
            x, tx = self._broadcast_operand(
                op.operands[1],
                env,
                axes.get(op.operands[1], ()),
                indent,
                env[op.result],
            )
            y, ty = self._broadcast_operand(
                op.operands[2],
                env,
                axes.get(op.operands[2], ()),
                indent,
                env[op.result],
            )
            self.lines.append(
                f'{indent}{r} = intent.select {c}, {x}, {y} : '
                f'({tc}, {tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "gather":
            d, i = (self.ssa[o] for o in op.operands)
            td, ti = (self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            mode = a.get("mode", "along_axis")
            # A bare IntentFunction has no Layer-B proof channel.  The caller
            # may pass only the exact projection derived from canonical J;
            # otherwise the target-specific authorization is never invented.
            candidate_refs = {ref for ref in (node_ref, op.node_id) if ref is not None}
            bounds_attr = (
                ", intent.index_bounds_proven = true"
                if candidate_refs & self.proven_index_bounds
                else ""
            )
            data_rank = len(env[op.operands[0]])
            index_rank = len(env[op.operands[1]])
            if axis < 0:
                axis += data_rank
            if mode == "along_axis" and index_rank == data_rank - 1:
                # JSON IntentIR permits the selected axis to be omitted from
                # index/result (one index per coordinate of the other axes).
                # Legalize to the stricter MLIR Intent op by inserting a unit
                # axis, gather at equal rank, then squeeze that unit axis.
                if index_rank == 0:
                    raise NotImplementedError(
                        "emit_mlir compressed along_axis gather does not yet "
                        "support a rank-0 index/result"
                    )
                expanded_index_shape = _insert_broadcast_axes(
                    env[op.operands[1]], (axis,)
                )
                assert expanded_index_shape is not None
                expanded_index, expanded_index_type = self._broadcast_operand(
                    op.operands[1], env, (axis,), indent
                )
                expanded_result_type = self._ttn_shape(
                    op.result, expanded_index_shape
                )
                self._n += 1
                gathered = f"%{op.result}_gather_expanded_{self._n}"
                self.lines.append(
                    f'{indent}{gathered} = intent.gather {d}, {expanded_index} '
                    f'{{axis = {axis} : i64, mode = "{mode}"{bounds_attr}}} : '
                    f'({td}, {expanded_index_type}) -> {expanded_result_type}')

                source_positions = [
                    dim for dim in range(data_rank) if dim != axis
                ]
                reassociation: list[list[int]] = []
                for index, position in enumerate(source_positions):
                    start = 0 if index == 0 else position
                    end = (
                        source_positions[index + 1]
                        if index + 1 < len(source_positions)
                        else data_rank
                    )
                    reassociation.append(list(range(start, end)))
                groups = "[" + ", ".join(
                    "[" + ", ".join(str(value) for value in group) + "]"
                    for group in reassociation
                ) + "]"
                self.lines.append(
                    f"{indent}{r} = tensor.collapse_shape {gathered} {groups} : "
                    f"{expanded_result_type} into {self._ttn(op.result, env)}"
                )
            else:
                if mode == "along_axis" and index_rank != data_rank:
                    raise NotImplementedError(
                        "emit_mlir along_axis gather requires index rank equal "
                        "to data rank or data rank minus one"
                    )
                self.lines.append(
                    f'{indent}{r} = intent.gather {d}, {i} '
                    f'{{axis = {axis} : i64, mode = "{mode}"{bounds_attr}}} : '
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
        elif op.op == "transpose":
            operand = op.operands[0]
            x = self.ssa[operand]
            perm = _fold_permutation(a["perm"], fold, extents or {})
            rendered = ", ".join(str(value) for value in perm)
            self.lines.append(
                f'{indent}{r} = intent.transpose {x} '
                f'{{perm = [{rendered}]}} : '
                f'({self._ttn(operand, env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "concat":
            xs = ", ".join(self.ssa[o] for o in op.operands)
            txs = ", ".join(self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            self.lines.append(
                f'{indent}{r} = intent.concat {xs} {{axis = {axis} : i64}} : '
                f'({txs}) -> {self._ttn(op.result, env)}')
        elif op.op == "pad":
            x = self.ssa[op.operands[0]]
            lo = ", ".join(
                str(_resolve_int(v, extents)) for v in a["lo"][fold:]
            )
            hi = ", ".join(
                str(_resolve_int(v, extents)) for v in a["hi"][fold:]
            )
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
            # Integer column-index vector; print its declared integer width.
            dt = a.get("dtype", "i32")
            shape = env[op.result]
            if len(shape) != 1:
                raise ValueError("iota result must resolve to rank 1")
            n = int(shape[0])
            declared_n = _resolve_int(a["n"], extents)
            if declared_n != n:
                raise ValueError(
                    f"iota attrs.n={declared_n} differs from result extent {n}"
                )
            if self.dtypes[op.result] != _normalize_dtype(dt, "iota dtype"):
                raise ValueError(
                    f"iota result dtype {self.dtypes[op.result]} differs from {dt}"
                )
            self.lines.append(
                f'{indent}{r} = intent.iota : tensor<{n}x{dt}>')
        elif op.op == "rand":
            # Counter-based uniform[0,1) RNG (C5); no operands, shape+dtype in the
            # result type, seed in the attr.
            self.lines.append(
                f'{indent}{r} = intent.rand {{seed = {int(a["seed"])} : i64}} : '
                f'{self._ttn(op.result, env)}')
        elif op.op == "const":
            # A literal value -> standard arith.constant (no intent semantics to
            # lower). A scalar splats dense over the (possibly rank-0) result
            # tensor; a list value (iota/arange materialized as an explicit
            # vector) prints as a dense elements array. An i1/int-dtyped const
            # (a boolean sentinel / mask threshold) carries its own element type.
            val = a["value"]
            dt = a.get("dtype", "f32")
            if dt in ("i1", "bool"):
                truthy = val not in (0, 0.0, False, None)
                dense = f"dense<{'true' if truthy else 'false'}>"
            elif dt in ("i8", "i16", "i32", "i64"):
                if isinstance(val, (list, tuple)):
                    dense = "dense<[" + ", ".join(str(int(v)) for v in val) + "]>"
                else:
                    dense = f"dense<{int(val)}>"
            elif isinstance(val, (list, tuple)):
                dense = "dense<[" + ", ".join(
                    _mlir_dense_float(v, dt) for v in val
                ) + "]>"
            else:
                dense = f"dense<{_mlir_dense_float(val, dt)}>"
            self.lines.append(
                f'{indent}{r} = arith.constant {dense} : {self._ttn(op.result, env)}')
        else:
            raise NotImplementedError(f"emit_mlir: unhandled op {op.op!r}")


def _tile(
    loop: IntentOp, extents: dict[str, int], schedule_tile: int | None
) -> int:
    if schedule_tile is not None:
        if isinstance(schedule_tile, bool) or not isinstance(schedule_tile, int):
            raise TypeError("loop schedule tile must be an integer")
        if schedule_tile <= 0:
            raise ValueError("loop schedule tile must be positive")
        return schedule_tile
    # Without C, use one full semantic-axis iteration.  This is a deterministic
    # semantics-preserving fallback, not a reconstruction of source tiling.
    if loop.trip is not None and loop.trip.axis in extents:
        return int(extents[loop.trip.axis])
    return 1


def _setup(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int,
    schedule_tile: int | None = None,
):
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
    tile = _tile(loop, extents, schedule_tile)
    scan_specs = []
    for s in loop.scan:
        ax = int(s["axis"]) - fold
        slshape = list(env[s["from"]])
        slshape[ax] = tile
        env[s["name"]] = slshape
        scan_specs.append((s["name"], s["from"], ax))
    carries = [c.body_argument for c in loop.carry]
    yields = [c.next_value for c in loop.carry]
    for state in loop.carry:
        if state.type is not None:
            if state.type.dtype.casefold().removeprefix("torch.") not in {
                "f32", "float32", "float",
            }:
                raise NotImplementedError(
                    "emit_mlir semantic-loop prototype currently supports "
                    f"f32 carry values, got {state.type.dtype!r}"
                )
            full = _resolve(state.type.shape, extents)
            env[state.body_argument] = full[fold:]
    for _ in range(len(loop.body) + 3):
        _infer_body(loop.body, env, fold, extents)
        moved = False
        for cname, yname in zip(carries, yields):
            if yname in env and env.get(cname) != env[yname]:
                env[cname] = list(env[yname])
                moved = True
        if not moved:
            break
    for state in loop.carry:
        if state.body_argument in env:
            env[state.result_value] = list(env[state.body_argument])
    return loop, env, carries, yields, scan_specs, tile


def emit_body_function(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 2,
    func_name: str = "flash_body_fwd",
    *,
    schedule_tile: int | None = None,
) -> str:
    """The loop body as an executable ``func.func`` (scan tiles + carries in,
    next carries out). This is what the host-driven trip loop launches."""
    loop, env, carries, yields, scan_specs, _ = _setup(
        fn, dims, fold, schedule_tile
    )
    dtypes = _loop_dtypes(fn, loop)
    p = _Printer(dtypes)

    referenced: set[str] = set()
    for op in loop.body:
        referenced.update(op.operands)
    # Outer inputs the body reads (Q, scale, ...): function inputs that appear in
    # the body directly (the scanned tensors enter via their slices, below).
    outer = [t.name for t in fn.inputs if t.name in referenced]

    args: list[str] = []
    for nm in outer:
        args.append(f"{p.arg(nm)}: {p._ttn(nm, env)}")
    for sname, _src, _ax in scan_specs:
        args.append(f"{p.arg(sname)}: {p._ttn(sname, env)}")
    for c in carries:
        args.append(f"{p.arg(c)}: {p._ttn(c, env)}")
    rty = ", ".join(p._ttn(c, env) for c in carries)

    for op in loop.body:
        p.emit_op(op, env, fold, extents=dict(dims))
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(p._ttn(y, env) for y in yields)
    symbol = _sanitize_mlir_symbol(func_name)

    head = (f"// Generated by intentir.frontend.emit_mlir from "
            f"{fn.name} (single-head, fold={fold}).\n"
            f"// One K-tile of the FlashAttention online-softmax recurrence.\n"
            f"module {{\n"
            f"  func.func @{symbol}({', '.join(args)}) -> ({rty}) {{")
    tail = (f"    return {yvals} : {ytys}\n  }}\n}}\n")
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


def emit_loop_module(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 2,
    func_name: str = "flash_fwd",
    *,
    schedule_tile: int | None = None,
) -> str:
    """The whole function with the structured ``intent.loop``. Representability +
    lowers to ``scf.for`` via ``-convert-intent-to-linalg``."""
    loop, env, carries, yields, scan_specs, tile = _setup(
        fn, dims, fold, schedule_tile
    )
    dtypes = _loop_dtypes(fn, loop)
    p = _Printer(dtypes)

    # Function signature: the IntentIR function inputs (folded, scalars rank-0).
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
    # Carry inits as function args too (so the module is self-contained and the
    # init *values* — NEG_INF/0/zeros — stay the caller's responsibility).
    init_names = [f"{c}_init" for c in carries]
    for c, init in zip(carries, init_names):
        env[init] = env[c]
        dtypes[init] = dtypes[c]
        args.append(f"{p.arg(init)}: {p._ttn(init, env)}")

    scan_ssa = ", ".join(p.ssa[s["from"]] for s in loop.scan)
    scan_tys = ", ".join(p._ttn(s["from"], env) for s in loop.scan)
    init_ssa = ", ".join(p.ssa[i] for i in init_names)
    init_tys = ", ".join(p._ttn(c, env) for c in carries)
    scan_axes = ", ".join(f"{int(s['axis']) - fold} : i64" for s in loop.scan)
    res_tys = ", ".join(p._ttn(c, env) for c in carries)

    # Body region block args: scan slices then carries.
    bargs = []
    for sname, _src, _ax in scan_specs:
        bargs.append(f"{p.arg(sname)}: {p._ttn(sname, env)}")
    for c in carries:
        # carry block arg shadows the init binding for the body
        bargs.append(f"{p.arg(c)}: {p._ttn(c, env)}")
    body_lines: list[str] = []
    saved, p.lines = p.lines, body_lines
    for op in loop.body:
        p.emit_op(
            op, env, fold, indent="      ", extents=dict(dims)
        )
    p.lines = saved
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(p._ttn(y, env) for y in yields)

    out_res = next((o for o in fn.body if o.op != "loop"), None)
    loop_res = p.fresh("loop")
    symbol = _sanitize_mlir_symbol(func_name)

    lines = [
        f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
        f"(single-head, fold={fold}).",
        "module {",
        f"  func.func @{symbol}({', '.join(args)}) -> ({res_tys}) {{",
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


def emit_module(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
    value_dimension_equalities: Sequence[tuple[int, int]] = (),
    output_values: Mapping[str, str] | None = None,
    inplace_outputs: Sequence[int] = (),
    paired_rotation: tuple[str, tuple[int, int, int, int], tuple[int, ...]]
    | None = None,
    loop_schedule_tile: int | None = None,
) -> str:
    """A flat (loop-free) IntentFunction as an executable ``func.func``.

    Inputs become function args (clean SSA names so the orchestrator keys them by
    JSON name), the atomic op-graph is emitted in order, and the declared outputs
    are returned. Covers softmax / layernorm / rmsnorm / silu / gelu / matmul /
    elementwise — every kernel whose body is a straight-line tensor graph.
    """
    extents = dict(dims)
    env: dict[str, list[int]] = {}
    for t in fn.inputs:
        env[t.name] = _resolve(t.shape, extents)[fold:]
    authored_return_values = {item.output: item.value for item in fn.returns}
    if len(authored_return_values) != len(fn.returns):
        raise ValueError("emit_module: duplicate logical return output")
    if output_values is None:
        return_values = authored_return_values
    else:
        return_values = dict(output_values)
        if any(
            output in authored_return_values
            and authored_return_values[output] != value
            for output, value in return_values.items()
        ):
            raise ValueError(
                "emit_module: canonical output binding conflicts with logical return"
            )
    unknown_return_outputs = sorted(
        set(return_values) - {item.name for item in fn.outputs}
    )
    if unknown_return_outputs:
        raise ValueError(
            "emit_module: logical returns reference undeclared outputs "
            f"{unknown_return_outputs}"
        )
    # A reduce whose result is a declared external output may intentionally
    # collapse its reduced axes (e.g. row_sum [M,N] -> [M]).  Seed only output
    # contracts; intermediate reductions retain the existing keepdims default.
    for t in fn.outputs:
        output_shape = _resolve(t.shape, extents)[fold:]
        env[t.name] = output_shape
        returned_value = return_values.get(t.name, t.name)
        previous_shape = env.get(returned_value)
        if previous_shape is not None and previous_shape != output_shape:
            raise ValueError(
                f"emit_module: return value {returned_value!r} has conflicting "
                "logical output shapes"
            )
        env[returned_value] = output_shape
    _infer_body(
        fn.body,
        env,
        fold,
        extents,
        broadcast_axes,
        loop_schedule_tile,
    )

    dtypes = _flat_dtypes(fn)
    p = _Printer(
        dtypes,
        broadcast_axes,
        proven_index_bounds,
        loop_schedule_tile,
    )
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
    for index, op in enumerate(fn.body):
        p.emit_op(
            op,
            env,
            fold,
            extents=extents,
            node_ref=f"body[{index}]",
        )

    outs: list[tuple[str, str]] = []
    for t in fn.outputs:
        value = return_values.get(t.name, t.name)
        if value not in p.ssa:
            raise ValueError(
                f"emit_module: output {t.name!r} return value {value!r} was never produced"
            )
        if value not in env or env[value] != env[t.name]:
            raise ValueError(
                f"emit_module: output {t.name!r} shape {env[t.name]} does not "
                f"match return value {value!r} shape {env.get(value)}"
            )
        if value not in dtypes or dtypes[value] != dtypes[t.name]:
            raise ValueError(
                f"emit_module: output {t.name!r} dtype {dtypes[t.name]} does not "
                f"match return value {value!r} dtype {dtypes.get(value)}"
            )
        outs.append((t.name, value))
    ovals = ", ".join(p.ssa[value] for _output, value in outs)
    # Respect i1 (comparison) / integer (iota) result element types so the
    # function signature matches the value actually produced.
    otys = ", ".join(p._ttn(output, env) for output, _value in outs)
    name = _sanitize_mlir_symbol(fn.name if func_name is None else func_name)
    function_attrs = _function_attributes(
        fn,
        value_dimension_equalities=value_dimension_equalities,
        inplace_outputs=inplace_outputs,
        paired_rotation=paired_rotation,
    )

    head = (f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
            f"(flat graph, fold={fold}).\n"
            f"module {{\n"
            f"  func.func @{name}({', '.join(args)}) -> ({otys})"
            f"{function_attrs} {{")
    tail = f"    return {ovals} : {otys}\n  }}\n}}\n"
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


__all__ = ["emit_body_function", "emit_loop_module", "emit_module"]
