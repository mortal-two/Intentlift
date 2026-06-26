#ifndef INTENT_CONVERSION_PASSES
#define INTENT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertIntentToLinalg : Pass<"convert-intent-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower intent atomic ops to linalg/tensor/arith on tensors.";
  let description = [{
    Lowers Layer-A atomic ops of the `intent` dialect to standard MLIR
    (linalg/tensor/arith), the first leg of the back-end spine
    intent -> linalg -> {scf,vector} -> gpu -> nvvm -> cubin. Starts with the
    elementwise `intent.map_binary` (equal-shape, no broadcast); reduce /
    contraction / broadcast follow.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // INTENT_CONVERSION_PASSES
