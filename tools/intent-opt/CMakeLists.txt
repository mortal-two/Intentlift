get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(intent-opt intent-opt.cpp)
llvm_update_compile_flags(intent-opt)
add_dependencies(intent-opt
  IntentConversionCommonPassIncGen
  IntentConversionCUDAPassIncGen
  IntentTargetCUDAPassIncGen
)

target_link_libraries(intent-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIntent
  MLIRIntentConversionCommon
  MLIRIntentConversionCUDA
  MLIRIntentTargetCUDA
  MLIROptLib
)

mlir_check_all_link_libraries(intent-opt)
