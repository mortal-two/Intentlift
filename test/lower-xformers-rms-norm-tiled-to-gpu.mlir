// RUN: intent-opt -convert-intent-row-reduction-to-gpu="block-threads=4 tile-cols=2" %s | FileCheck %s

// xformers _rms_norm_kernel, the LARGE-N tiled+smem path (M5b): N/T = 16/4 = 4
// columns per thread exceeds tile-cols=2, so the per-thread feature strip is
// split into num_tiles=2 tiles of 2 columns. Each `reduce` becomes ONE scf.for
// over the tiles accumulating a per-thread partial (iter_args), with the row
// cached into a workgroup (dynamic shared) buffer on the way; the normalize
// epilogue reads the row back from that cache (NOT a 2nd HBM read of X) — this
// turns the library's explicit two-X-read loop into ONE X read.
//
// The block reduction is a power-of-two shared-memory tree over a scratch slice
// of the SAME dynamic buffer (CHECK-NOT gpu.all_reduce): its lowering would emit
// a separate static `.shared` buffer that ptxas packs INSIDE the demoted
// dynamic-shared cache and clobbers it. The frontier (here X) is the only cached
// value -> ONE [N] f32 cache.
//
// CHECK-LABEL: func.func @xf_rms_norm(
// CHECK-SAME:    %{{.*}}: memref<2x16xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<2x16xf32>)
// CHECK:         gpu.launch
// CHECK-SAME:      dynamic_shared_memory_size
// CHECK:         %[[RAW:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
// CHECK:         memref.view %[[RAW]]{{.*}} to memref<16xf32, #gpu.address_space<workgroup>>
// the reduction pass: one scf.for accumulating a partial, caching X to smem
// CHECK:         scf.for %{{.*}} iter_args(%{{.*}}) -> (f32) {
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xf32>
// CHECK:           memref.store %{{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
// CHECK:           scf.yield %{{.*}} : f32
// the block reduction is a shared-mem tree (NO gpu.all_reduce static buffer)
// CHECK-NOT:     gpu.all_reduce
// CHECK:         gpu.barrier
// CHECK:         math.rsqrt
// the epilogue: read the row back from the smem cache, not from %arg0 (X)
// CHECK:         scf.for
// CHECK:           memref.load %view[%{{.*}}] : memref<16xf32, #gpu.address_space<workgroup>>
// CHECK:           memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x16xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @xf_rms_norm(%X: tensor<2x16xf32>, %W: tensor<16xf32>, %eps: tensor<f32>)
    -> tensor<2x16xf32> {
  %sq = intent.map_binary %X, %X {fn = "mul"} : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %ms = intent.reduce %sq {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x16xf32>) -> tensor<2x1xf32>
  %mse = intent.map_binary %ms, %eps {fn = "add"} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mse {fn = "rsqrt"} : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %Xn = intent.map_binary %X, %rstd {fn = "mul"} : (tensor<2x16xf32>, tensor<2x1xf32>) -> tensor<2x16xf32>
  %Y = intent.map_binary %Xn, %W {fn = "mul"} : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  return %Y : tensor<2x16xf32>
}
