add_mlir_library(MLIRIntentConversionCommon
  ConvertIntentToLinalg.cpp

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
  MLIRSCFDialect
)

add_subdirectory(CUDA)

# Preserve the historical aggregate target while allowing CPU and future
# targets to link only the target-neutral Intent -> Linalg conversion.
add_library(MLIRIntentConversion ALIAS MLIRIntentConversionCUDA)
