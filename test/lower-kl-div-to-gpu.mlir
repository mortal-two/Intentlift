// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=4 cross-row-finalize=true' %s | FileCheck %s --check-prefix=FINALIZE

// kl_div (batchmean, log_target=False): the cross-row AtomicOut fold. Per row,
// loss_v = y_true * (log(max(y_true, eps)) - y); the per-row sum is reduced over
// the feature axis (gpu.all_reduce add), then a CROSS-ROW axis-0 sum folds every
// block's per-row scalar into ONE rank-0 loss scalar via a thread-0-guarded
// memref.atomic_rmw addf (host pre-zeroes). The batchmean /BT scale is a uniform
// post-reduce factor folded into each block's contribution (linear over the
// sum). This fuses the host `output.sum()/BT` the library launches separately.
// math.log lowers (downstream) to lg2.approx.f32 * ln2, NOT a __nv_logf extern.
// M=4, N=8, T=4. Single result -> M/N derived from the Row inputs.
// CHECK-LABEL: func.func @kl_div(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK:         math.log
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         arith.cmpi eq
// CHECK:         scf.if
// CHECK:         memref.atomic_rmw addf %{{.*}}, %{{.*}}[] : (f32, memref<f32>) -> f32
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
// FINALIZE-LABEL: func.func @kl_div(
// FINALIZE-SAME: intent.backend.reduction_topology = "two_stage_cross_row_finalize"
// FINALIZE-SAME: intent.backend.stage_count = 2 : i64
// FINALIZE: memref.alloc() : memref<4xf32>
// FINALIZE: gpu.launch
// FINALIZE: memref.store {{.*}}[%{{.*}}] : memref<4xf32>
// FINALIZE: gpu.terminator
// FINALIZE: gpu.launch
// FINALIZE: gpu.all_reduce{{.*}}add
// FINALIZE: memref.store {{.*}}[] : memref<f32>
// FINALIZE-NOT: memref.atomic_rmw
// FINALIZE: memref.dealloc
// FINALIZE-NOT: intent.
func.func @kl_div(%y: tensor<4x8xf32>, %gt: tensor<4x8xf32>,
                  %eps: tensor<f32>, %invBT: tensor<f32>) -> tensor<f32> {
  %me = intent.map_binary %gt, %eps {fn = "max"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %lg = intent.map_unary %me {fn = "log"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %d  = intent.map_binary %lg, %y {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %le = intent.map_binary %gt, %d {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %rowsum = intent.reduce %le {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %total = intent.reduce %rowsum {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<4xf32>) -> tensor<f32>
  %loss = intent.map_binary %total, %invBT {fn = "mul"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %loss : tensor<f32>
}
