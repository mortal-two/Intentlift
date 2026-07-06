add_mlir_library(MLIRIntentConversion
  ConvertIntentToLinalg.cpp
  ConvertIntentRowReductionToGPU.cpp
  ConvertIntentElementwiseToGPU.cpp
  ConvertIntentGatherToGPU.cpp
  ConvertIntentLayoutToGPU.cpp
  ConvertIntentRngToGPU.cpp
  ConvertIntentArgReduceToGPU.cpp
  ConvertIntentSortToGPU.cpp
  ConvertIntentPoolToGPU.cpp
  ConvertIntentUpsampleToGPU.cpp
  ConvertIntentPolarToGPU.cpp
  ConvertIntentConvToGPU.cpp
  ConvertIntentContractionToGPU.cpp
  ConvertIntentContractionToGPUMMA.cpp
  ConvertIntentContractionToGPUMMAWS.cpp
  ConvertIntentLoopContractionToGPU.cpp
  ConvertIntentFlashMMAToGPU.cpp
  ConvertIntentAttentionBackwardToGPU.cpp
  ConvertIntentSplitKReduceToGPU.cpp
  ConvertIntentSplitKReduceBackwardToGPU.cpp

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
  MLIRNVGPUDialect
  MLIRVectorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
)
