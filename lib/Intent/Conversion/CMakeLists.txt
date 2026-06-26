add_mlir_library(MLIRIntentConversion
  ConvertIntentToLinalg.cpp
  ConvertIntentRowReductionToGPU.cpp
  ConvertIntentElementwiseToGPU.cpp
  ConvertIntentContractionToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Intent/Conversion

  DEPENDS
  IntentConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIntent
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransformUtils
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRFuncDialect
)
