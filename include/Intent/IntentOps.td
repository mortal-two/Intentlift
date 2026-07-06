#ifndef INTENT_OPS
#define INTENT_OPS

include "IntentDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tensor-level atomic ops (Layer A semantics, Layer B constraints as attrs).
// Starter set: map_binary / reduce / contraction. More ops (scan, gather,
// scatter, loop, ...) follow the same pattern (see spec/ir/atomic-ops.md).
//===----------------------------------------------------------------------===//

def Intent_MapBinaryOp : Intent_Op<"map_binary"> {
  let summary = "Elementwise binary map (Layer A).";
  let description = [{
    Elementwise binary op over two ranked tensors. `fn` selects the
    pointwise operation (add/sub/mul/div/max/min/pow/cmp). The comparison fns
    (gt/ge/lt/le/eq/ne) produce an `i1`-typed result tensor (feeding `intent.select`);
    every other fn is element-type-preserving.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, StrAttr:$fn);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_SelectOp : Intent_Op<"select"> {
  let summary = "Elementwise select (Layer A): where(cond, x, y).";
  let description = [{
    Per-element `cond ? x : y` (numpy `where`). `cond` is an `i1`-typed ranked
    tensor; `x`/`y`/result share one element type. Shapes broadcast numpy-style
    (e.g. an `[M,N]` `i1` condition selecting between two rank-0 scalars), so the
    verifier constrains element types only, not shapes. Owned by M3; reused by
    M4/M6/M7/M9 (do not re-add). Lowered via `arith.select` over the scalar
    condition produced by a `map_binary` comparison fn.
  }];
  let arguments = (ins AnyRankedTensor:$cond, AnyRankedTensor:$x,
                       AnyRankedTensor:$y);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$cond `,` $x `,` $y attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_IotaOp : Intent_Op<"iota", [Pure]> {
  let summary = "Column-index vector (Layer A): [0, 1, ..., N-1].";
  let description = [{
    Materializes the rank-1 integer index vector `0,1,...,N-1` (numpy
    `arange`), where `N` is the static extent of the result tensor. The
    canonical source of a per-column position used to build a one-hot / gather
    condition (`iota == target`) for cross_entropy's target-logit pick. No
    operands; the result type encodes both `N` and the integer width. In a
    block-cooperative lowering the value for column `col` resolves to `col`
    itself (an `arith.index_cast` of the thread's column index) — there is no
    materialized vector. Owned by M6; reused wherever a column index is needed.
  }];
  let arguments = (ins);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Intent_MapUnaryOp : Intent_Op<"map_unary"> {
  let summary = "Elementwise unary map (Layer A).";
  let description = [{
    Elementwise unary op over one ranked tensor. `fn` selects the pointwise
    operation. Most fns are shape- AND element-type-preserving
    (exp/log/neg/abs/sqrt/rsqrt/recip/tanh/sigmoid/cos/sin/tan/exp2/erf), the same
    class the transcendental peephole turns into NVVM approx intrinsics.
    Predicate fns (`isnan`) instead produce an `i1`-typed result tensor (the
    boolean condition for `intent.select`, mirroring the `map_binary` comparison
    fns); `logical_not` is the `i1`->`i1` complement. Shape-preserving throughout.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_CastOp : Intent_Op<"cast"> {
  let summary = "Elementwise dtype cast (Layer A).";
  let description = [{
    Element-type conversion (numpy `astype`) of one ranked tensor to `to_dtype`.
    Shape-preserving; only the element type changes. Lowers to the matching
    `arith` conversion for the (source, target) pair — `truncf`/`extf` between
    float widths, `fptosi`/`sitofp` across the float/int boundary, `trunci`/
    `extsi` between integer widths — and folds to an identity alias when source
    and target element types are already equal (the f32-only corpus case).
    The M3b-deferred op, landed by C0 so the dtype-convert (`cast`/`to`) family
    extracts + lowers.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$to_dtype);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ScanOp : Intent_Op<"scan"> {
  let summary = "Inclusive cumulative scan along one axis (Layer A).";
  let description = [{
    Inclusive cumulative scan of `input` along `axis` with `fn` (currently
    `sum` = cumsum). Shape-preserving: the result has the same shape and element
    type as the input; `result[..., i, ...] = combine(input[..., 0..i, ...])`.
    Unlike `intent.reduce` (which collapses the axis to a single value) and
    `intent.loop` (which yields only the final carry), scan keeps the per-element
    running value — the primitive sparsemax's `cumsum` of the sorted row needs.
    Owned by M4. In a block-cooperative lowering it lowers to a single-block
    inclusive scan over a contiguous column-to-thread tiling (per-thread
    sequential prefix + a cross-thread exclusive scan of the segment totals
    through `gpu.dynamic_shared_memory` + `gpu.barrier`).
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn, I64Attr:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_GatherOp : Intent_Op<"gather"> {
  let summary = "Data-dependent gather along one axis (Layer A).";
  let description = [{
    Gather elements of `data` at runtime positions read from an integer `index`
    tensor along `axis`. Two `mode`s cover the data-dependent-addressing family:

    * `"along_axis"` (torch.gather / numpy `take_along_axis`): `index` has the
      same rank as `data`, the result is shaped like `index`, and
      `result[i..] = data[i.. with i[axis] replaced by index[i..]]`.
    * `"take"` (index_select / embedding row-gather / numpy `take`): `index` is a
      rank-r selector inserted at `axis`; the result rank is
      `data.rank - 1 + index.rank` and
      `result[pre.., k.., post..] = data[pre.., index[k..], post..]`. An
      embedding is exactly `take` with a rank-1 `[N]` index into axis 0 of the
      `[V,D]` weight (`result[n,d] = weight[index[n], d]`).

    The C0/C1 cluster's new mechanism: the load address is a RUNTIME value (read
    from `index`), not an affine function of the iteration — the GPU lowering
    emits a data-dependent `memref.load` at the index-computed coordinate. The
    result element type equals `data`'s; `index` must be integer-typed.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$index,
                       I64Attr:$axis, StrAttr:$mode);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $index attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ScatterOp : Intent_Op<"scatter"> {
  let summary = "Data-dependent scatter/scatter-add along one axis (Layer A).";
  let description = [{
    Write `src` elements into a copy of `data` at runtime positions read from an
    integer `index` tensor along `axis` (torch.scatter / scatter_add_ /
    numpy `put_along_axis`). `mode` is `"along_axis"`: `index` and `src` share a
    shape, and for each element `out[i.. with i[axis]=index[i..]] (op)= src[i..]`
    where the combine `op` is selected by `reduce`:

    * `"none"` — overwrite (torch.scatter, nondeterministic for duplicate
      indices, matching torch's contract);
    * `"add"`  — accumulate (torch.scatter_add_, order-independent).

    The result has `data`'s shape and element type. The GPU lowering treats it as
    an in-place update of the (caller-cloned) `data` buffer: a data-dependent
    `memref.store` (reduce `"none"`) or `memref.atomic_rmw addf` (reduce
    `"add"`, reusing the M4/M9 atomic -> `red.global.add.f32` on sm_120a) at the
    index-computed coordinate. `index` must be integer-typed.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$index,
                       AnyRankedTensor:$src, I64Attr:$axis, StrAttr:$mode,
                       StrAttr:$reduce);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $index `,` $src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Structural layout ops (Layer A). Pure DATA MOVEMENT: every output element is a
// source element chosen by a STATIC structural rule (an affine index / a
// piecewise-by-offset select) or a pad constant — no arithmetic. Only
// broadcast/reshape were implicit before; these make concat/pad/tile/flip/triu
// (cat/stack/hstack/vstack, pad, repeat/tile, flip, triu) first-class. Lowered by
// `convert-intent-layout-to-gpu`: one block per output row, cols over threads,
// each output element's SOURCE coord computed by the rule then `memref.load`ed
// (or set to the pad constant). See spec/ir/atomic-ops.md (C4).
//===----------------------------------------------------------------------===//

def Intent_ConcatOp : Intent_Op<"concat"> {
  let summary = "Variadic concatenation along one axis (Layer A).";
  let description = [{
    Place `inputs` end-to-end along `axis` at their cumulative offsets
    (numpy `concatenate` / torch.cat). All inputs share the result's rank and
    element type and agree on every non-`axis` extent; the result `axis` extent is
    the sum of the input `axis` extents. `hstack`/`vstack` are concat on axis 1/0;
    `stack` is concat after an implicit unsqueeze of each input at `axis`.

    The one genuinely-new bit vs. the affine layout ops is the VARIADIC + offset
    placement: `out[coord]` is owned by exactly one input, selected piecewise by
    the cumulative offsets, and loaded from that input at the offset-shifted coord.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs, I64Attr:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_PadOp : Intent_Op<"pad"> {
  let summary = "Constant pad along every axis (Layer A).";
  let description = [{
    Pad `input` with `lo[d]` elements before and `hi[d]` after along each dim `d`,
    filling the border with the constant `value` (numpy `pad` mode="constant" /
    torch `constant_pad_nd`). Result extent `= input[d] + lo[d] + hi[d]`.
    `out[coord] = in-bounds(coord) ? input[coord - lo] : value` — a piecewise
    select on the per-dim in-bounds predicate plus the pad constant.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$lo,
                       I64ArrayAttr:$hi, F64Attr:$value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_TileOp : Intent_Op<"tile"> {
  let summary = "Tile (repeat the whole tensor) along every axis (Layer A).";
  let description = [{
    Repeat `input` `reps[d]` times along each dim `d` (numpy `tile` / torch.tile /
    torch.Tensor.repeat). Result extent `= input[d] * reps[d]` and
    `out[coord] = input[coord mod input_extent]` — a per-dim affine `remui` back
    into the source, no guard (every output coord maps to a valid source).
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$reps);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_FlipOp : Intent_Op<"flip"> {
  let summary = "Reverse the element order along the given axes (Layer A).";
  let description = [{
    Reverse `input` along each dim in `axes` (numpy `flip` / torch.flip).
    Shape-preserving; `out[.., i, ..] = input[.., N-1-i, ..]` for each flipped
    axis (extent `N`) — a per-dim affine `N-1-i` remap of the source coord.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_TriuOp : Intent_Op<"triu"> {
  let summary = "Upper-triangular mask above the k-th diagonal (Layer A).";
  let description = [{
    Zero the elements below the `k`-th diagonal of a rank-2 `input` (numpy `triu` /
    torch.triu): `out[i,j] = (j >= i + k) ? input[i,j] : 0`. Shape-preserving; a
    select on the iota-derived (i,j) position over the always-in-bounds load.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$k);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Stateful seeded RNG (Layer A). A counter-based, STATELESS-per-element philox
// hash: the value at output element linear-index `i` is a pure function of
// (seed, i) — no visited-order dependence, no global RNG state. `intent.rand`
// produces uniform[0,1) f32; every other RNG op (dropout / randn / normal /
// uniform(a,b) / *_like) COMPOSES rand with the existing map/select primitives
// (dropout = select(rand >= p, x/(1-p), 0); randn = Box-Muller of two rands;
// uniform(a,b) = a + (b-a)*rand). Lowered by `convert-intent-rng-to-gpu`: one
// block per output row, each element computes philox4x32-10(seed, row*Co+col)
// -> uniform -> the composed elementwise transform -> store. See spec/ir (C5).
//===----------------------------------------------------------------------===//

def Intent_RandOp : Intent_Op<"rand"> {
  let summary = "Counter-based uniform[0,1) RNG (Layer A, stateless per element).";
  let description = [{
    Materialize a tensor of uniform[0,1) f32 samples whose element at row-major
    linear index `i` is `philox4x32-10(seed, i)` mapped to `[0,1)` — a PURE
    function of the `seed` attribute and the element index (no operands, no
    global RNG state, order-independent), so two launches with the same seed are
    bit-identical and different seeds decorrelate. The seed splits into the two
    philox key words; the counter is `(i, 0, 0, 0)`; the first output word gives
    the 24-bit-mantissa uniform. The result must be f32; its shape is carried in
    the result type (like `intent.iota`'s extent). rand_like is rand at the
    input's shape; randn / normal / uniform(a,b) / dropout are COMPOSITIONS of
    rand with map_unary / map_binary / select (no new op).
  }];
  let arguments = (ins I64Attr:$seed);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Intent_ReduceOp : Intent_Op<"reduce"> {
  let summary = "Reduction along axes (Layer A semantics + Layer B associativity).";
  let description = [{
    Reduce `input` along `axes` with `fn` (sum/max/min/prod/any/all).
    `assoc` records reduction associativity for verification:
    `associative` (bit-exact) or `float_nondeterministic` (tolerance).
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       StrAttr:$fn,
                       I64ArrayAttr:$axes,
                       StrAttr:$assoc);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ContractionOp : Intent_Op<"contraction"> {
  let summary = "Einsum-style contraction (Layer A).";
  let description = [{
    General tensor contraction with einsum-style `subscripts`
    (e.g. "bhqd,bhkd->bhqk"). Covers matmul / batched-matmul / dot.
  }];
  let arguments = (ins AnyRankedTensor:$a, AnyRankedTensor:$b, StrAttr:$subscripts);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$a `,` $b attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Structured loop with loop-carried state (Layer A recurrence + Layer C tiling).
// FlashAttention's K-loop: carries the online-softmax state (running max, sum,
// accumulator) while scanning K/V tiles. See spec/ir/structured-loop.md.
//===----------------------------------------------------------------------===//

def Intent_YieldOp : Intent_Op<"yield",
    [Pure, Terminator, HasParent<"LoopOp">]> {
  let summary = "Terminator of an intent.loop body; yields the next carries.";
  let description = [{
    Yields the loop-carried values for the next trip iteration (and, after the
    last iteration, the loop results). Operand count/types match the parent
    `intent.loop` carries.
  }];
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";
  // No-operand builder so SingleBlockImplicitTerminator can materialize the
  // implicit terminator (mirrors scf.yield).
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Intent_LoopOp : Intent_Op<"loop",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Structured loop with loop-carried state and scanned inputs.";
  let description = [{
    A structured loop (an explicit induction variable, loop-carried `inits`, and
    a single-block body region) that scans `scan_inputs` tile-by-tile along their
    `scan_axes` while threading carried state. This is the IntentIR primitive for
    FlashAttention's K-loop: `inits` = (running max m, running sum l, accumulator
    Oacc); each trip slices a tile (`tile_hint` rows, Layer C) of K and V; the
    body recomputes the online-softmax update and `intent.yield`s the new carries.

    Body block arguments are the per-trip scan slices (one per `scan_inputs`, in
    order) followed by the current carries (one per `inits`). Results are the
    final carries; result types equal `inits` types (carry types are invariant).

    Layer attribution: the recurrence is Layer A (algorithmic intent); `tile_hint`
    is a non-binding Layer C schedule hint (the back-end may retune it).
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$scan_inputs,
                       Variadic<AnyType>:$inits,
                       I64ArrayAttr:$scan_axes,
                       OptionalAttr<I64Attr>:$tile_hint);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `scan` `(` ($scan_inputs^ `:` type($scan_inputs))? `)`
    `init` `(` ($inits^ `:` type($inits))? `)`
    attr-dict (`->` type($results)^)? `do` $body
  }];
}

#endif // INTENT_OPS
