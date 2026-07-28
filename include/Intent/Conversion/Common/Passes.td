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
