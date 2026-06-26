set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentTarget)
add_public_tablegen_target(IntentTargetPassIncGen)
