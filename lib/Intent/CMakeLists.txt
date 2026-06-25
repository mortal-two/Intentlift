add_mlir_dialect_library(MLIRIntent
  IntentDialect.cpp
  IntentOps.cpp
  Transforms/ConvertIntentToLinalg.cpp
  Transforms/LowerGpuMathToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Intent

  DEPENDS
  IntentIncGen
  IntentPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransformUtils
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRLLVMDialect
)
