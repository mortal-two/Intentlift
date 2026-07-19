// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// A regular contiguous Col[N] parameter follows the same vec4 column mapping
// as Row[M,N]. Retype and preload it as vector<4xf32>; indexed/ColSeg gathers
// are deliberately outside this structural fast path.
// CHECK-LABEL: func.func @rms_col_vec4(
// CHECK-SAME: %{{[^,]+}}: memref<2x8xvector<4xf32>>, %{{[^,]+}}: memref<8xvector<4xf32>>, %{{[^,]+}}: memref<f32>, %{{[^)]+}}: memref<2x8xvector<4xf32>>)
// CHECK-SAME: intent.launch_plan = {{.*}}vector_width = 4
// CHECK-COUNT-2: memref.load %{{.*}}[%{{.*}}] : memref<8xvector<4xf32>>
// CHECK: vector.extract
// CHECK-NOT: memref.load %{{.*}}[%{{.*}}] : memref<32xf32>
// CHECK: vector.from_elements
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x8xvector<4xf32>>
// CHECK-NOT: intent.
func.func @rms_col_vec4(
    %X: tensor<2x32xf32>, %W: tensor<32xf32>, %eps: tensor<f32>)
    -> tensor<2x32xf32> {
  %square = intent.map_binary %X, %X {fn = "mul"}
      : (tensor<2x32xf32>, tensor<2x32xf32>) -> tensor<2x32xf32>
  %mean = intent.reduce %square {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x32xf32>) -> tensor<2x1xf32>
  %mean_eps = intent.map_binary %mean, %eps {fn = "add"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mean_eps {fn = "rsqrt"}
      : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %scaled = intent.map_binary %X, %rstd {fn = "mul"}
      : (tensor<2x32xf32>, tensor<2x1xf32>) -> tensor<2x32xf32>
  %Y = intent.map_binary %scaled, %W {fn = "mul"}
      : (tensor<2x32xf32>, tensor<32xf32>) -> tensor<2x32xf32>
  return %Y : tensor<2x32xf32>
}
