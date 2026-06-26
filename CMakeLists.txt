cmake_minimum_required(VERSION 3.20)
project(IntentIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Out-of-tree MLIR dialect project for IntentIR.
# Requires a system / prebuilt MLIR (tested against MLIR 20.1.8 in
# /usr/lib/llvm-20). Configure with:
#   -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
#   -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/Intent)
add_subdirectory(include/Target)
add_subdirectory(lib/Intent)
add_subdirectory(lib/Target)
add_subdirectory(tools/intent-opt)
add_subdirectory(test)
