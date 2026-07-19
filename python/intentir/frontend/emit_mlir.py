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
executable form). Scalar inputs (``scale``) are emitted as rank-0 tensors while
retaining their declared element type.
"""

from __future__ import annotations

import ast
import json
import operator
import re

from intentir.ir.model import IntentFunction, IntentOp

_INVALID_MLIR_SYMBOL_CHAR = re.compile(r"[^A-Za-z0-9_]")


def _mlir_string(value: str) -> str:
    """Encode UTF-8 text as one MLIR string literal without lossy escaping."""

    encoded = value.encode("utf-8")
    chunks: list[str] = []
    for byte in encoded:
        if 0x20 <= byte <= 0x7E and byte not in (0x22, 0x5C):
            chunks.append(chr(byte))
        else:
            chunks.append(f"\\{byte:02X}")
    return '"' + "".join(chunks) + '"'


def _validated_scalar_equalities(fn: IntentFunction) -> list[dict[str, object]]:
    """Return machine-derived integer scalar equalities carried by Layer B.

    Only assumptions emitted inside a validated closed-interval bounds proof
    are eligible. Free-text contracts and ordinary observed-value notes are not
    specialization facts. ``emit_p4`` strips unbound proof objects before this
    function runs, so diagnostic/unbound JSON cannot create the typed attr.
    """

    inputs = {
        tensor.name: (index, tensor)
        for index, tensor in enumerate(fn.inputs)
    }
    equalities: dict[str, int] = {}
    for op in fn.body:
        proof = op.layer_b.get("index_bounds_proof")
        if not isinstance(proof, dict):
            continue
        assumptions = proof.get("assumptions")
        if not isinstance(assumptions, list):
            continue
        for assumption in assumptions:
            if (
                not isinstance(assumption, dict)
                or assumption.get("kind")
                != "validated_layer_b_integer_equality"
            ):
                continue
            name = assumption.get("input")
            value = assumption.get("equals")
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or name not in inputs
            ):
                continue
            _index, tensor = inputs[name]
            if tensor.dtype not in {"i8", "i16", "i32", "i64"} or tuple(
                tensor.shape
            ) != ():
                continue
            previous = equalities.get(name)
            if previous is not None and previous != value:
                raise ValueError(
                    f"conflicting validated scalar equality for {name}: "
                    f"{previous} versus {value}"
                )
            equalities[name] = value
    return [
        {
            "input": name,
            "argument_index": inputs[name][0],
            "dtype": inputs[name][1].dtype,
            "equals": equalities[name],
        }
        for name in sorted(equalities, key=lambda item: inputs[item][0])
    ]


def _semantic_contract_attributes(fn: IntentFunction) -> str:
    """Persist Layer-B obligations and Layer-C hints as real MLIR attrs.

    Intent operations already encode the executable Layer-A graph.  These
    canonical JSON string attributes keep the remaining two layers attached to
    the same verified function so selector/audit code can consume them without
    consulting prose or reconstructing hints from a kernel name.
    """

    layer_b = [
        {
            "index": index,
            "op": op.op,
            "result": op.result,
            "layer_b": op.layer_b,
        }
        for index, op in enumerate(fn.body)
        if op.layer_b
    ]
    b_json = json.dumps(
        layer_b, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    c_json = json.dumps(
        fn.layer_c, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    attrs = [
        "intent_layer_b_json = " + _mlir_string(b_json),
        "intent_layer_c_json = " + _mlir_string(c_json),
    ]
    scalar_equalities = _validated_scalar_equalities(fn)
    if scalar_equalities:
        flattened = [
            str(value)
            for item in scalar_equalities
            for value in (item["argument_index"], item["equals"])
        ]
        attrs.append(
            "intent.scalar_equalities = array<i64: "
            + ", ".join(flattened)
            + ">"
        )

    # A validated conditional writeback is an explicit destination alias, not
    # a fresh mathematical output.  Materialize the existing ``intent.inplace``
    # ABI contract only when the Layer-B record is unambiguous: the result is a
    # declared output, the mutation is single-writer/non-atomic, and exactly one
    # select operand is a shape/dtype-compatible function input.  The C++ row
    # lowering can then preserve the source's conditional store instead of
    # writing an identity copy of an entire row.
    input_index = {tensor.name: index for index, tensor in enumerate(fn.inputs)}
    output_index = {tensor.name: index for index, tensor in enumerate(fn.outputs)}
    input_type = {
        tensor.name: (tensor.dtype, tuple(tensor.shape)) for tensor in fn.inputs
    }
    output_type = {
        tensor.name: (tensor.dtype, tuple(tensor.shape)) for tensor in fn.outputs
    }
    inplace = [-1] * len(fn.outputs)
    for op in fn.body:
        if (
            op.op != "select"
            or not isinstance(op.result, str)
            or op.result not in output_index
        ):
            continue
        mutation = op.layer_b.get("mutation")
        aliases = op.layer_b.get("aliasing")
        conflict = op.layer_b.get("conflict_contract")
        if (
            not isinstance(mutation, dict)
            or mutation.get("kind")
            not in {"conditional_writeback", "in_place_conditional_writeback"}
            or not isinstance(aliases, dict)
            or aliases.get("no_cross_argument_aliases") is not True
            or not isinstance(conflict, dict)
            or conflict.get("atomics_required") is not False
            or conflict.get("single_writer_per_element") is not True
        ):
            continue
        compatible_inputs = [
            operand
            for operand in op.operands
            if operand in input_index
            and input_type[operand] == output_type[op.result]
        ]
        if len(compatible_inputs) == 1:
            inplace[output_index[op.result]] = input_index[compatible_inputs[0]]

    # Direct source mutation is also an ABI alias when Layer B names the exact
    # input and output.  This covers kernels such as xFormers RMSNormAdd, where
    # ``x_ptr`` is overwritten with ``X + Y`` and returned as ``X_out``.  Keep
    # the rule deliberately typed and fail closed: vague storage prose, an
    # undeclared input/output, a mismatched result, or an incompatible tensor
    # type cannot create an alias.
    direct_inplace_kinds = {
        "in_place_overwrite",
        "inplace_overwrite",
        "overwrite_in_place",
    }
    partial_inplace_kinds = {"partial_writeback"}
    for op in fn.body:
        mutation = op.layer_b.get("mutation")
        if not isinstance(mutation, dict):
            continue
        kind = mutation.get("kind")
        if kind in direct_inplace_kinds:
            input_name = mutation.get("input")
            output_name = mutation.get("output")
        elif kind in partial_inplace_kinds:
            input_name = mutation.get("preserve_unwritten_from")
            output_name = op.result
        else:
            continue
        if (
            not isinstance(op.result, str)
            or not isinstance(input_name, str)
            or not isinstance(output_name, str)
            or output_name != op.result
            or input_name not in input_index
            or output_name not in output_index
            or input_type[input_name] != output_type[output_name]
        ):
            continue
        result_index = output_index[output_name]
        alias_index = input_index[input_name]
        previous = inplace[result_index]
        if previous >= 0 and previous != alias_index:
            raise ValueError(
                f"conflicting inplace contract for output {output_name}: "
                f"input {previous} versus {alias_index}"
            )
        inplace[result_index] = alias_index
    if any(index >= 0 for index in inplace):
        attrs.append(
            "intent.inplace = array<i64: "
            + ", ".join(str(index) for index in inplace)
            + ">"
        )

    # Materialize correctness-relevant Layer-B facts as typed attributes only
    # when the validated JSON states them explicitly.  The C++ selector can
    # then use the fact without parsing prose or guessing from a symbol name.
    # A missing/vague relationship deliberately produces no fast-path attr.
    for op in fn.body:
        if op.op != "scan":
            continue
        relationships = op.layer_b.get("input_relationships")
        if not isinstance(relationships, dict):
            continue
        sorted_x = relationships.get("SortedX")
        if not isinstance(sorted_x, dict) or sorted_x.get("descending") is not True:
            continue
        attrs.append("intent_sorted_descending = true")
        matches = sorted_x.get("matches_host_sort_of")
        if isinstance(matches, str) and matches:
            attrs.append(
                "intent_sorted_matches_host_sort_of = " + _mlir_string(matches)
            )
        timed = sorted_x.get("host_preprocessing_in_kernel_timing")
        if isinstance(timed, bool):
            attrs.append(
                "intent_host_preprocessing_in_kernel_timing = "
                + ("true" if timed else "false")
            )
        break
    return " attributes {" + ", ".join(attrs) + "}"


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
        axes = [int(a) - fold for a in op.attrs["axes"]]
        keepdims = list(shp)
        for ax in axes:
            keepdims[ax] = 1
        collapsed = [dim for index, dim in enumerate(shp) if index not in axes]
        declared = env.get(op.result)
        if declared is None and op.out is not None:
            declared = env.get(op.out)
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
        return _resolve([op.attrs["n"]], extents or {})
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
            "logical_and", "logical_or", "logical_xor", "isclose"}
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


def _layer_b(op: IntentOp) -> dict[str, object]:
    """Return Layer-B obligations, accepting one legacy nested representation."""

    merged: dict[str, object] = {}
    nested = op.attrs.get("layer_b")
    if isinstance(nested, dict):
        merged.update(nested)
    merged.update(op.layer_b)
    return merged


def _gather_bounds_proven(op: IntentOp) -> bool:
    """Return only machine-readable, validated Layer-B bounds claims."""

    layer_b = _layer_b(op)
    # Contract identifiers are descriptive LLM text, not proof objects.  In
    # particular, an arbitrary ``*_gather_indices_in_bounds`` suffix must not
    # become an unsafe-load authorization.  P2 may expose a structured fact
    # only after its validator has established it.
    return layer_b.get("index_in_bounds") is True


def _declared_op_dtype(op: IntentOp) -> str | None:
    if not isinstance(op.type, dict) or "dtype" not in op.type:
        return None
    return _normalize_dtype(op.type["dtype"], f"{op.op} result type")


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
        inferred = [_normalize_dtype(a.get("to_dtype"), "cast to_dtype")]
    elif op.op == "contraction":
        layer_b = _layer_b(op)
        accumulator = layer_b.get(
            "accumulator_dtype", layer_b.get("accumulation_dtype")
        )
        # An invalid/missing obligation is not allowed to force a dtype.  Fall
        # back to the operands' natural common type instead.
        dtype = (
            accumulator
            if isinstance(accumulator, str) and accumulator in _FLOAT_DTYPES
            else None
        )
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
        for name, dtype in _infer_op_dtypes(op, dtypes).items():
            changed |= _record_dtype(dtypes, name, dtype, op.op)
        if op.out is not None and isinstance(op.result, str):
            result_dtype = dtypes.get(op.result)
            if result_dtype is not None:
                changed |= _record_dtype(
                    dtypes, op.out, result_dtype, f"{op.op} destination"
                )
    return changed


def _flat_dtypes(fn: IntentFunction) -> dict[str, str]:
    dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"input {tensor.name!r}")
        for tensor in fn.inputs
    }
    _infer_body_dtypes(fn.body, dtypes)
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


def _infer_body(body: list[IntentOp], env: dict[str, list[int]], fold: int,
                extents: dict[str, int] | None = None) -> None:
    for _ in range(len(body) + 2):
        changed = False
        # Structural consumers can constrain an otherwise ambiguous producer.
        # Most importantly, a declared output after transpose determines the
        # transpose operand rank/shape, which lets an intermediate reduce choose
        # collapsed vs keepdims without relying on a model-specific heuristic.
        for op in reversed(body):
            if (
                op.op != "transpose"
                or not isinstance(op.result, str)
                or op.result not in env
                or len(op.operands) != 1
            ):
                continue
            operand = op.operands[0]
            result_shape = env[op.result]
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
    def __init__(self, dtypes: dict[str, str]) -> None:
        self.lines: list[str] = []
        self._n = 0
        self.ssa: dict[str, str] = {}   # JSON name -> SSA name
        self.dtypes = dtypes

    def _ttn(self, name: str, env: dict[str, list[int]]) -> str:
        """Tensor type for one JSON value from the inferred dtype environment."""
        if name not in self.dtypes:
            raise ValueError(f"emit_mlir: no dtype inferred for {name!r}")
        return _tt(env[name], self.dtypes[name])

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
                extents: dict[str, int] | None = None) -> None:
        a = op.attrs
        if op.op == "cast":
            to = _normalize_dtype(a.get("to_dtype"), "cast to_dtype")
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
        elif op.op == "gather":
            d, i = (self.ssa[o] for o in op.operands)
            td, ti = (self._ttn(o, env) for o in op.operands)
            axis = int(a["axis"]) - fold
            mode = a.get("mode", "along_axis")
            bounds_attr = (
                ", intent.index_bounds_proven = true"
                if _gather_bounds_proven(op)
                else ""
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
            # `n` may be a symbolic runtime extent. Shape inference has already
            # resolved it into the result environment for this specialization.
            n = env[op.result][0]
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


def emit_loop_module(fn: IntentFunction, dims: dict[str, int], fold: int = 2,
                     func_name: str = "flash_fwd") -> str:
    """The whole function with the structured ``intent.loop``. Representability +
    lowers to ``scf.for`` via ``-convert-intent-to-linalg``."""
    loop, env, carries, yields, scan_specs, tile = _setup(fn, dims, fold)
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
    # A reduce whose result is a declared external output may intentionally
    # collapse its reduced axes (e.g. row_sum [M,N] -> [M]).  Seed only output
    # contracts; intermediate reductions retain the existing keepdims default.
    for t in fn.outputs:
        env[t.name] = _resolve(t.shape, extents)[fold:]
    _infer_body(fn.body, env, fold, extents)

    dtypes = _flat_dtypes(fn)
    p = _Printer(dtypes)
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
    for op in fn.body:
        p.emit_op(op, env, fold, extents=extents)
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
    name = _sanitize_mlir_symbol(fn.name if func_name is None else func_name)

    contract_attrs = _semantic_contract_attributes(fn)
    head = (f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
            f"(flat graph, fold={fold}).\n"
            f"module {{\n"
            f"  func.func @{name}({', '.join(args)}) -> ({otys})"
            f"{contract_attrs} {{")
    tail = f"    return {ovals} : {otys}\n  }}\n}}\n"
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


__all__ = ["emit_body_function", "emit_loop_module", "emit_module"]
