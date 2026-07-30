// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=32 tile-cols=128 rows-per-block=32 promote-rank0-scalars=false defer-epilogue-col-loads=false' %s | FileCheck %s

// One warp owns each row and 32 warps share a 1024-thread CTA.  tile-cols=128
// keeps a 4096-wide row in the register path; this exact schedule coordinate is
// what makes the already-implemented subgroup-rows topology reachable for
// production normalization shapes.
// CHECK-LABEL: func.func @rms_norm_multirow(
// CHECK-SAME: intent.backend.block_threads = 1024 : i64
// CHECK-SAME: intent.backend.reduction_topology = "subgroup_rows"
// CHECK-SAME: intent.backend.rows_per_block = 32 : i64
// CHECK: %[[GRID:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK:.*]] = arith.constant 1024 : index
// CHECK: gpu.launch{{.*}}%[[GRID]]{{.*}}%[[BLOCK]]
// CHECK: gpu.shuffle xor
// CHECK-NOT: gpu.barrier
func.func @rms_norm_multirow(
    %X: tensor<32x4096xf32>, %W: tensor<4096xf32>, %eps: tensor<f32>)
    -> tensor<32x4096xf32> {
  %x2 = intent.map_binary %X, %X {fn = "mul"}
    : (tensor<32x4096xf32>, tensor<32x4096xf32>) -> tensor<32x4096xf32>
  %mean = intent.reduce %x2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<32x4096xf32>) -> tensor<32x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<32x1xf32>, tensor<f32>) -> tensor<32x1xf32>
  %inv = intent.map_unary %me {fn = "rsqrt"}
    : (tensor<32x1xf32>) -> tensor<32x1xf32>
  %xs = intent.map_binary %X, %inv {fn = "mul"}
    : (tensor<32x4096xf32>, tensor<32x1xf32>) -> tensor<32x4096xf32>
  %Y = intent.map_binary %xs, %W {fn = "mul"}
    : (tensor<32x4096xf32>, tensor<4096xf32>) -> tensor<32x4096xf32>
  return %Y : tensor<32x4096xf32>
}
