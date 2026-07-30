// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=128 promote-uniform-scalars=true' %s | FileCheck %s --check-prefix=HOST
// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=128 promote-uniform-scalars=false' %s | FileCheck %s --check-prefix=DEVICE

// HOST-LABEL: func.func @scale(
// HOST-SAME: intent.backend.host_value_capture_count = 1 : i64
// HOST-SAME: "cuda.b.uniform_host_value_capture.v1"
// HOST: memref.load
// HOST: gpu.launch
// HOST-NOT: intent.

// DEVICE-LABEL: func.func @scale(
// DEVICE-SAME: intent.backend.host_value_capture_count = 0 : i64
// DEVICE: gpu.launch
// DEVICE: memref.load
// DEVICE-NOT: intent.
func.func @scale(%x: tensor<2x127xf32>, %alpha: tensor<1xf32>)
    -> tensor<2x127xf32> {
  %out = intent.map_binary %x, %alpha {fn = "mul"}
    : (tensor<2x127xf32>, tensor<1xf32>) -> tensor<2x127xf32>
  return %out : tensor<2x127xf32>
}
