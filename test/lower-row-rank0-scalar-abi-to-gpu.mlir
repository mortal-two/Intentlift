// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=128 promote-rank0-scalars=true' %s | FileCheck %s --check-prefix=HOST
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=128 promote-rank0-scalars=false' %s | FileCheck %s --check-prefix=DEVICE

// The scalar ABI policy is a real MLIR schedule axis.  Both candidates consume
// the same typed graph and B contract: one captures eps by value from the host
// stub, the other retains the rank-0 load inside gpu.launch.

// HOST-LABEL: func.func @rms(
// HOST-SAME: intent.backend.host_value_capture_count = 1 : i64
// HOST-SAME: "cuda.b.rank0_host_value_capture.v1"
// HOST: memref.load
// HOST: gpu.launch
// HOST-NOT: intent.

// DEVICE-LABEL: func.func @rms(
// DEVICE-SAME: intent.backend.host_value_capture_count = 0 : i64
// DEVICE: gpu.launch
// DEVICE: memref.load
// DEVICE-NOT: intent.
func.func @rms(%x: tensor<2x127xf32>, %eps: tensor<f32>)
    -> tensor<2x127xf32> {
  %square = intent.map_binary %x, %x {fn = "mul"}
    : (tensor<2x127xf32>, tensor<2x127xf32>) -> tensor<2x127xf32>
  %mean = intent.reduce %square
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<2x127xf32>) -> tensor<2x1xf32>
  %shifted = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %inv = intent.map_unary %shifted {fn = "rsqrt"}
    : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %out = intent.map_binary %x, %inv {fn = "mul"}
    : (tensor<2x127xf32>, tensor<2x1xf32>) -> tensor<2x127xf32>
  return %out : tensor<2x127xf32>
}
