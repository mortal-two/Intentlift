add_mlir_library(MLIRIntentTargetCUDA
  LowerBf16MmaToNVVM.cpp
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
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRLLVMCommonConversion
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRNVGPUDialect
  MLIRNVGPUToNVVM
  MLIRNVVMDialect
  MLIRSCFDialect
  MLIRSCFTransforms
)
