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
