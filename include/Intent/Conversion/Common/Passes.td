#ifndef INTENT_CONVERSION_COMMON_PASSES
#define INTENT_CONVERSION_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def ConvertIntentToLinalg : Pass<"convert-intent-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower intent atomic ops to linalg/tensor/arith on tensors.";
  let description = [{
    Lowers Layer-A atomic ops of the `intent` dialect to target-neutral standard
    MLIR (linalg/tensor/arith/scf). Elementwise `intent.map_binary` and
    `intent.select` use affine input maps for scalar, right-aligned and size-1
    broadcast; explicit semantic rank insertion is legalized by the frontend
    before this pass. Target-specific scheduling and code generation belong to
    a separate backend pipeline.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"reassociateFloatReductions", "reassociate-float-reductions",
           "bool", "false",
           "Attach only the reassoc fast-math flag to sum/mean/prod combines "
           "whose intent.reduce explicitly declares assoc = "
           "float_nondeterministic. Other floating-point operations and "
           "ordered reductions remain strict.">,
  ];
}

def ExpandRsqrtForLibm : Pass<"expand-rsqrt-for-libm", "::mlir::ModuleOp"> {
  let summary = "Expand math.rsqrt for host libm lowering.";
  let description = [{
    Rewrites `math.rsqrt(x)` to `1 / math.sqrt(x)` so a subsequent
    `convert-math-to-libm` uses only the standard sqrt/sqrtf host ABI instead
    of the non-standard rsqrtf symbol. This is a host-pipeline compatibility
    pass; target-neutral and GPU pipelines must not schedule it.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
  ];
}

def TileLinalgByIteratorKind : Pass<
    "tile-linalg-by-iterator-kind", "::mlir::ModuleOp"> {
  let summary = "Tile Linalg operations with independent parallel/reduction extents.";
  let description = [{
    Applies the upstream TilingInterface/SCF tiler to tensor-semantics Linalg
    operations. Parallel and reduction iterators receive independent static
    tile extents; zero leaves that iterator kind untiled. The pass is target
    neutral and records only actual tiled operations/loops. Optional producer
    fusion starts from terminal consumers and stops at multi-use read-once
    frontiers instead of duplicating producer computation. RVV uses it for real
    retile and structural-path search, but no target or C semantics are embedded
    here.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"parallelTile", "parallel-tile", "unsigned", "0",
           "Tile extent for every parallel iterator (zero disables)">,
    Option<"reductionTile", "reduction-tile", "unsigned", "0",
           "Tile extent for every reduction iterator (zero disables)">,
    Option<"fuseProducers", "fuse-producers", "bool", "false",
           "Tile terminal consumers and fuse single-consumer tensor producers into their SCF loops">,
  ];
}

def VectorizeLinalgRowReduction : Pass<
    "vectorize-linalg-row-reduction", "::mlir::ModuleOp"> {
  let summary = "Build full-row vector accumulators for reassociable Linalg reductions.";
  let description = [{
    Matches tensor-semantics `linalg.generic` operations with one or more
    leading parallel iterators followed by one reduction iterator, an identity
    input map, a projection output map, and either an f32 add combiner carrying
    the explicit reassoc flag or an associative f32 maxnumf combiner. The
    reduction region may contain an arbitrary
    Elementwise scalar producer payload before the final add; those operations
    are cloned lane-wise using their MLIR traits. The pass strip-mines the
    reduction axis, keeps one vector accumulator live across the complete row,
    and emits exactly one matching add/max horizontal finalize per logical row.
    `scalable-vector` derives the strip size from `vector.vscale`; otherwise
    `vector-width` is fixed. Tail reads use transfer padding and lower to masked
    accesses. `joint-reductions` traces pure identity-mapped elementwise producer
    DAGs to a common root and evaluates two or more independent row statistics
    in one traversal with one root packet load and multiple vector accumulators.
    It can also rewrite a full-domain centered variance into joint raw moments
    when a materialized scalar equality proves that the mask bound equals the
    complete feature extent; without that hard fact the masked graph is rejected.
    Matching is structural and target/source-name blind. Unsupported graphs
    remain untouched;
    `require-vectorization` makes a requested Full candidate fail closed when no
    operation is transformed.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", "4",
           "Fixed lanes, or minimum scalable lanes, in the vector accumulator">,
    Option<"scalableVector", "scalable-vector", "bool", "false",
           "Use a scalable vector accumulator and a vector.vscale-derived strip size">,
    Option<"jointReductions", "joint-reductions", "bool", "false",
           "Fuse independent same-root row reductions into one vector traversal">,
    Option<"requireJointReductions", "require-joint-reductions", "bool", "false",
           "Fail the pass if no same-root joint-reduction group is transformed">,
    Option<"requireVectorization", "require-vectorization", "bool", "false",
           "Fail the pass if no structurally supported row reduction is vectorized">,
  ];
}

def VectorizeLinalgMixedPrecisionDotReduction : Pass<
    "vectorize-linalg-mixed-precision-dot-reduction", "::mlir::ModuleOp"> {
  let summary = "Vectorize f16-input, f32-accumulating last-axis dot reductions.";
  let description = [{
    Structurally matches a tensor-semantics Linalg reduction with two identity-
    mapped f16 inputs, leading parallel iterators, one final reduction iterator,
    explicit f16-to-f32 extension, multiplication, and a reassociable f32 add.
    It emits strip-mined vector f16 loads, vector widening conversions, an f32
    vector accumulator, and one horizontal finalize per row.  The pass is
    source-name blind and preserves the original output tensor, including any
    downstream padding operation.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", "4",
           "Fixed lanes, or minimum scalable lanes, in the widening accumulator">,
    Option<"scalableVector", "scalable-vector", "bool", "true",
           "Use scalable vectors and a vector.vscale-derived strip size">,
    Option<"requireVectorization", "require-vectorization", "bool", "false",
           "Fail when no structurally supported mixed-precision dot is transformed">,
  ];
}

def FuseLinalgSoftmax : Pass<
    "fuse-linalg-softmax", "::mlir::ModuleOp"> {
  let summary = "Lower canonical stable row Softmax to one fused vector state.";
  let description = [{
    Structurally matches a rank-2 f32 stable Softmax graph lowered through
    Linalg: row max, broadcast subtract, exp, reassociable row sum, and
    broadcast divide. The fused hybrid state emits one vector max traversal,
    one scalar exp+live-sum traversal that writes directly into the final tensor
    scratch, and one vector in-place normalization traversal. This target-
    appropriate split avoids pretending libm exp is a native RVV vector op.
    Matching rejects ordered sums, extra users, non-canonical maps, and
    unsupported shapes.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", "4",
           "Fixed lanes, or minimum scalable lanes, in each row packet">,
    Option<"scalableVector", "scalable-vector", "bool", "false",
           "Use scalable vector packets and a vector.vscale-derived strip size">,
    Option<"requireFusion", "require-fusion", "bool", "false",
           "Fail the pass if no canonical stable row Softmax is fused">,
    Option<"approximateExp", "approximate-exp", "bool", "false",
           "Use a vector range-reduced polynomial exp and vector sum">,
  ];
}

def LowerRowMaskToLoops : Pass<
    "lower-row-mask-to-loops", "::mlir::ModuleOp"> {
  let summary = "Lower a typed row-mask broadcast graph to row-hoisted loops.";
  let description = [{
    Structurally matches the bufferized form of cast(i1[M])->f32 followed by
    pointwise data[M,N] * mask[M].  The replacement loads the mask once per row
    and keeps the feature traversal contiguous.  With zero-fill enabled, false
    rows write zero without reading data; that mode changes NaN/Inf multiplied
    by zero and therefore belongs only to an execution-domain checked search
    state.  Matching is source-name and target-name blind.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"zeroFill", "zero-fill", "bool", "false",
           "For false rows, write zero without loading the data tensor">,
    Option<"requireLowering", "require-lowering", "bool", "false",
           "Fail when no exact typed row-mask graph is lowered">,
    Option<"vectorWidth", "vector-width", "unsigned", "1",
           "Explicit contiguous packet width for each inner-row iteration">,
  ];
}

def LowerPairedRotationToLoops : Pass<
    "lower-paired-rotation-to-loops", "::mlir::ModuleOp"> {
  let summary = "Lower a closed interleaved paired rotation to direct loops.";
  let description = [{
    Consumes the target-neutral paired-rotation view emitted only after the
    accepted J proves the two in-place even/odd rotations and their exact angle
    layout.  After bufferization, the exploded gather/cast/scatter graph is
    replaced by direct SCF/memref loops.  The contiguous state keeps the pair
    axis innermost; the head-angle-reuse state hoists each angle pair across all
    query and key heads.  A fixed-vector state loads one contiguous 2W packet,
    deinterleaves it with vector.shuffle, and writes one contiguous result packet;
    the scalable state uses masked gather/scatter.  Matching is source-name and
    target-name blind.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"headAngleReuse", "head-angle-reuse", "bool", "false",
           "Hoist each cosine/sine pair across all query and key heads">,
    Option<"requireLowering", "require-lowering", "bool", "false",
           "Fail when no exact closed interleaved paired rotation is lowered">,
    Option<"vectorWidth", "vector-width", "unsigned", "1",
           "Number of interleaved pairs handled per vector iteration">,
    Option<"scalableVector", "scalable-vector", "bool", "false",
           "Use a vector.vscale-derived interleaved-pair packet">,
  ];
}

def LowerGatherChainToLoops : Pass<
    "lower-gather-chain-to-loops", "::mlir::ModuleOp"> {
  let summary = "Compose a typed two-level gather into direct indexed loads.";
  let description = [{
    Matches the target-neutral tensor graph
      take(data, row_index, axis=0) -> along_axis(tmp, col_index, axis=1)
    with the exact expand/collapse adapters emitted by Intent materialization.
    It replaces the materialized intermediate row tensor with direct
    data[row_index[i], col_index[i]] loads.  Vector mode emits vector.gather,
    which LLVM can lower to an RVV indexed load.  The matcher is source-name
    blind and rejects dynamic, non-contiguous, non-i32-index, or tailed shapes.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", "1",
           "Indexed elements issued by each loop iteration">,
    Option<"requireLowering", "require-lowering", "bool", "false",
           "Fail when no exact typed two-level gather is present">,
  ];
}

def FuseLinalgSiblingOutputs : Pass<
    "fuse-linalg-sibling-outputs", "::mlir::ModuleOp"> {
  let summary = "Fuse returned elementwise siblings that share an iteration domain.";
  let description = [{
    Matches two pure tensor-semantics linalg.generic operations returned as
    distinct public outputs.  When both have the same all-parallel iteration
    domain, write-only identity outputs, and compatible indexing maps, their
    input lists and scalar bodies are merged into one multi-output generic.
    Shared input tensors are consequently loaded once per element.  Matching is
    source-name blind and does not cross reductions, updates, or observable
    intermediates.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"requireFusion", "require-fusion", "bool", "false",
           "Fail if no exact returned sibling pair is fused">,
  ];
}

def ForwardBufferResultsToOutParams : Pass<
    "forward-buffer-results-to-out-params", "::mlir::ModuleOp"> {
  let summary = "Forward bufferized tensor results into explicit out params.";
  let description = [{
    After one-shot-bufferize and buffer-results-to-out-params, replace a
    function-local result allocation followed by one final memref.copy with
    direct writes to the equivalent out-parameter.  This removes allocator and
    copy overhead before CPU/RVV LLVM lowering.  The pass is structural and
    does not infer aliases: it only rewrites exact-type, single-copy result
    buffers whose destination is an entry block argument.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

def LowerDisjointMemrefCopyToLoops : Pass<
    "lower-disjoint-memref-copy-to-loops", "::mlir::ModuleOp"> {
  let summary = "Lower provably disjoint memref.copy operations to SCF loops.";
  let description = [{
    Expands ranked memref.copy operations whose source and destination have
    equal shapes and element types.  The memref.copy contract makes overlapping
    source/destination locations undefined, so a valid operation already
    provides the disjointness needed by the generated direct loop.  Layouts may
    differ because memref.load/store preserve each operand's addressing.  This
    removes the external memrefCopy runtime from standalone CPU/RVV objects,
    including copies between function ABI arguments.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // INTENT_CONVERSION_COMMON_PASSES
