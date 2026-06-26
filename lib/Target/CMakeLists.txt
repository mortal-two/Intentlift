add_mlir_library(MLIRIntentTarget
  LowerGpuMathToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Target

  DEPENDS
  IntentTargetPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  MLIRMathDialect
  MLIRLLVMDialect
)
