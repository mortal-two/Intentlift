// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=3 %s | FileCheck %s

// group_norm: layer_norm's mean+var (here the NON-centered E[X^2]-E[X]^2 form
// liger uses) over rows flattened as (batch, group) -> M = B*G, each row of
// hidden = (C//G)*spatial elements. The ONLY new thing vs layer_norm is the
// AFFINE W/B index: gamma/beta are Col[C] gathered at channel = group*CPG +
// col//SPATIAL, where group = bx % G. This is expressed GENERALLY as a ColSeg
// arg carrying an `intent.col_index_map` affine-map arg-attr; emitAt emits the
// arith floordiv/mul/add channel index (no per-kernel branch). Here B=2,G=2,
// CPG=2,SPATIAL=3 -> M=4, N=6, C=4; G recovered as C/CPG = 2 so group = bx % 2.
// CHECK-LABEL: func.func @group_norm(
// CHECK-SAME:    %{{.*}}: memref<4x6xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x6xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.remui %{{.*}}, %{{.*}} : index
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// the affine channel index: col floordiv SPATIAL + group*CPG, then load W/B[ch]
// CHECK:         arith.divui %{{.*}}, %{{.*}} : index
// CHECK:         arith.muli %{{.*}}, %{{.*}} : index
// CHECK:         arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         memref.load %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x6xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @group_norm(
    %X: tensor<4x6xf32>,
    %W: tensor<4xf32> {intent.col_index_map = affine_map<(d0, d1) -> (d1 floordiv 3 + d0 * 2)>},
    %B: tensor<4xf32> {intent.col_index_map = affine_map<(d0, d1) -> (d1 floordiv 3 + d0 * 2)>},
    %eps: tensor<f32>) -> (tensor<4x6xf32>, tensor<4x1xf32>, tensor<4x1xf32>) {
  %m = intent.reduce %X {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x6xf32>) -> tensor<4x1xf32>
  %xsq = intent.map_binary %X, %X {fn = "mul"} : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<4x6xf32>
  %exsq = intent.reduce %xsq {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x6xf32>) -> tensor<4x1xf32>
  %m2 = intent.map_binary %m, %m {fn = "mul"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
  %var = intent.map_binary %exsq, %m2 {fn = "sub"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
  %vare = intent.map_binary %var, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %vare {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xc = intent.map_binary %X, %m {fn = "sub"} : (tensor<4x6xf32>, tensor<4x1xf32>) -> tensor<4x6xf32>
  %xn = intent.map_binary %xc, %rstd {fn = "mul"} : (tensor<4x6xf32>, tensor<4x1xf32>) -> tensor<4x6xf32>
  %xnw = intent.map_binary %xn, %W {fn = "mul"} : (tensor<4x6xf32>, tensor<4xf32>) -> tensor<4x6xf32>
  %Y = intent.map_binary %xnw, %B {fn = "add"} : (tensor<4x6xf32>, tensor<4xf32>) -> tensor<4x6xf32>
  return %Y, %m, %rstd : tensor<4x6xf32>, tensor<4x1xf32>, tensor<4x1xf32>
}
