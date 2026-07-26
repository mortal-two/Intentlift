add_mlir_library(MLIRIntentConversionCommon
  ConvertIntentToLinalg.cpp
  ExpandRsqrtForLibm.cpp
  ForwardBufferResultsToOutParams.cpp
  LowerDisjointMemrefCopyToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Intent/Conversion

  DEPENDS
  IntentConversionCommonPassIncGen

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
  MLIRMathTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
)

add_subdirectory(CUDA)
