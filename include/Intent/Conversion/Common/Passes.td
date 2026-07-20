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

#endif // INTENT_CONVERSION_COMMON_PASSES
