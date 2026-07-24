add_mlir_library(MLIRIntentConversionCommon
  ConvertIntentToLinalg.cpp
  ExpandRsqrtForLibm.cpp

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
  MLIRSCFDialect
)

add_subdirectory(CUDA)
