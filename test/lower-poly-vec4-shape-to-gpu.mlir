// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// Formal PolyNorm uses K=N/T=16 and therefore the vec4 register-tile path.
// This reduced graph retains its scalar parameter gather plus compact concat
// output while exercising the same K=16 vector geometry.
// CHECK-LABEL: func.func @poly_vec4_shape
// CHECK-SAME: intent.launch_plan = {{.*}}vector_width = 4
// CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xvector<4xf32>>
// CHECK: vector.extract
// CHECK: vector.from_elements
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xvector<4xf32>>
// CHECK-COUNT-2: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32>
// CHECK-NOT: intent.
func.func @poly_vec4_shape(
    %X: tensor<2x64xf32>, %W: tensor<3xf32>, %eps: tensor<f32>)
    -> (tensor<2x64xf32>, tensor<2x2xf32>) {
  %zero = arith.constant dense<0> : tensor<i32>
  %w0 = intent.gather %W, %zero {axis = 0 : i64, mode = "take"} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
  %square = intent.map_binary %X, %X {fn = "mul"} : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
  %sum = intent.reduce %square {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x64xf32>) -> tensor<2x1xf32>
  %n = arith.constant dense<64.0> : tensor<f32>
  %mean = intent.map_binary %sum, %n {fn = "div"} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %mean_eps = intent.map_binary %mean, %eps {fn = "add"} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mean_eps {fn = "rsqrt"} : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %stats = intent.concat %mean, %rstd {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  %Y = intent.map_binary %X, %w0 {fn = "mul"} : (tensor<2x64xf32>, tensor<f32>) -> tensor<2x64xf32>
  return %Y, %stats : tensor<2x64xf32>, tensor<2x2xf32>
}
