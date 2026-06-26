// RUN: intent-opt -convert-intent-row-reduction-to-gpu="block-threads=4 tile-cols=2" %s | FileCheck %s

// xformers _rms_norm_add_kernel, tiled+smem with IN-PLACE residual (M5b):
//   A = X + residual   (the new residual, written back over X in place)
//   h1 = A * rsqrt(mean(A*A) + eps) * W
// The ONE row-varying carrier cached in workgroup shared is the post-add A
// (markFrontier folds X + residual into a single buffer — NOT two), so the
// reduction loop reads X and residual ONCE, computes A, caches A, and squares A
// for the partial; the epilogue reads A back from the cache (NOT a 2nd HBM read)
// to write BOTH the in-place A output and h1. The A output memref is passed the
// X pointer by the harness (in-place); read-once across the barrier makes the
// alias safe. Block reduction is a shared-mem tree, no gpu.all_reduce.
//
// CHECK-LABEL: func.func @xf_rms_norm_add(
// CHECK-SAME:    %{{.*}}: memref<2x16xf32>, %{{.*}}: memref<2x16xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<2x16xf32>, %{{.*}}: memref<2x16xf32>)
// CHECK:         gpu.launch
// CHECK-SAME:      dynamic_shared_memory_size
// CHECK:         gpu.dynamic_shared_memory
// ONE [N] f32 cache (the A frontier), not two (X and residual separately)
// CHECK:         memref.view %{{.*}} to memref<16xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.view %{{.*}} to memref<16xf32, #gpu.address_space<workgroup>>
// reduction loop: read X and residual, add -> A, cache A to smem
// CHECK:         scf.for %{{.*}} iter_args(%{{.*}}) -> (f32) {
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xf32>
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xf32>
// CHECK:           arith.addf
// CHECK:           memref.store %{{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
// CHECK:           scf.yield
// CHECK-NOT:     gpu.all_reduce
// CHECK:         math.rsqrt
// epilogue reads A back from the cache and writes both outputs from it
// CHECK:         scf.for
// CHECK:           memref.load %view[%{{.*}}] : memref<16xf32, #gpu.address_space<workgroup>>
// CHECK:           gpu.terminator
// CHECK-NOT:     intent.
func.func @xf_rms_norm_add(%X: tensor<2x16xf32>, %Resid: tensor<2x16xf32>,
                           %W: tensor<16xf32>, %eps: tensor<f32>)
    -> (tensor<2x16xf32>, tensor<2x16xf32>) {
  %A = intent.map_binary %X, %Resid {fn = "add"} : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %sq = intent.map_binary %A, %A {fn = "mul"} : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %ms = intent.reduce %sq {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x16xf32>) -> tensor<2x1xf32>
  %mse = intent.map_binary %ms, %eps {fn = "add"} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mse {fn = "rsqrt"} : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %An = intent.map_binary %A, %rstd {fn = "mul"} : (tensor<2x16xf32>, tensor<2x1xf32>) -> tensor<2x16xf32>
  %Y = intent.map_binary %An, %W {fn = "mul"} : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  return %A, %Y : tensor<2x16xf32>, tensor<2x16xf32>
}
