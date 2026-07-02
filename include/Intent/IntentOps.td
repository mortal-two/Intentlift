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
    (gt/ge/lt/le/eq) produce an `i1`-typed result tensor (feeding `intent.select`);
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
