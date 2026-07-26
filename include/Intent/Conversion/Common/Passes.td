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
    Expands memref.copy when its source is a fresh memref.alloc and source and
    destination have equal ranked shapes and element types.  Fresh allocation
    proves disjointness without an alias oracle, so the generated direct loop
    preserves copy semantics and removes the external memrefCopy runtime from
    standalone CPU/RVV objects.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // INTENT_CONVERSION_COMMON_PASSES
