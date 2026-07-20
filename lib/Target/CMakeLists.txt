add_subdirectory(CUDA)

# Compatibility name for existing tools; concrete target code is CUDA-only.
add_library(MLIRIntentTarget ALIAS MLIRIntentTargetCUDA)
