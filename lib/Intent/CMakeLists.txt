add_mlir_dialect_library(MLIRIntent
  IntentDialect.cpp
  IntentOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Intent

  DEPENDS
  IntentIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_subdirectory(Conversion)
