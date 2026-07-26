// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s

// Exact structural shape of FlashAttention `_bwd_preprocess_do_o_dot`:
// [1,M,1,D] f16 inputs are widened, multiplied, reduced on D, then the
// singleton [1,M,1] statistic is returned through the physical no-op
// [0,2,1] view as [1,1,M].  The source schedule launches 4 warps for D=64, so
// lanes 64..127 must remain in the collective with the sum identity while
// loading only a safe in-bounds address.
// CHECK-LABEL: func.func @bwd_preprocess_rank4(
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<128x64xf16>, %[[DO:[^:]+]]: memref<128x64xf16>, %[[DELTA:[^:]+]]: memref<128xf32>)
// CHECK-SAME: attributes {
// CHECK-SAME: intent.backend.folded_constexpr_select_count = 0 : i64
// CHECK-SAME: intent.backend.predicated_value_if_count = 0 : i64
// CHECK-SAME: intent.launch_plan = {block = [128, 1, 1]
// CHECK-SAME: smem = 16
// CHECK:       gpu.launch
// CHECK:       gpu.dynamic_shared_memory
// CHECK:       memref.view {{.*}} to memref<4xf32, #gpu.address_space<workgroup>>
// CHECK:       %[[ACTIVE:.*]] = arith.cmpi ult, %{{.*}}, %{{.*}} : index
// CHECK:       %[[SAFE:.*]] = arith.select %[[ACTIVE]], %{{.*}}, %{{.*}} : index
// CHECK:       %[[O16:.*]] = memref.load %[[OUT]][%{{.*}}, %[[SAFE]]] : memref<128x64xf16>
// CHECK:       %[[DO16:.*]] = memref.load %[[DO]][%{{.*}}, %[[SAFE]]] : memref<128x64xf16>
// CHECK:       %[[O32:.*]] = arith.extf %[[O16]] : f16 to f32
// CHECK:       %[[DO32:.*]] = arith.extf %[[DO16]] : f16 to f32
// CHECK:       %[[TERM:.*]] = arith.mulf %[[O32]], %[[DO32]] : f32
// CHECK:       %[[MASKED:.*]] = arith.select %[[ACTIVE]], %[[TERM]], %{{.*}} : f32
// CHECK-NOT:   gpu.all_reduce
// CHECK-COUNT-5: gpu.shuffle
// CHECK:       gpu.barrier
// CHECK-COUNT-2: gpu.shuffle
// CHECK-NOT:   gpu.barrier
// CHECK:       memref.store %{{.*}}, %[[DELTA]][%{{.*}}] : memref<128xf32>
// CHECK-NOT:   gpu.all_reduce
// CHECK-NOT:   intent.
func.func @bwd_preprocess_rank4(
    %Out: tensor<1x128x1x64xf16>,
    %DO: tensor<1x128x1x64xf16>) -> tensor<1x1x128xf32> {
  %out_f32 = intent.cast %Out {to_dtype = "f32"} : (tensor<1x128x1x64xf16>) -> tensor<1x128x1x64xf32>
  %do_f32 = intent.cast %DO {to_dtype = "f32"} : (tensor<1x128x1x64xf16>) -> tensor<1x128x1x64xf32>
  %dot_terms = intent.map_binary %out_f32, %do_f32 {fn = "mul"} : (tensor<1x128x1x64xf32>, tensor<1x128x1x64xf32>) -> tensor<1x128x1x64xf32>
  %delta_bmh = intent.reduce %dot_terms {fn = "sum", axes = [3], assoc = "associative"} : (tensor<1x128x1x64xf32>) -> tensor<1x128x1xf32>
  %Delta = intent.transpose %delta_bmh {perm = [0, 2, 1]} : (tensor<1x128x1xf32>) -> tensor<1x1x128xf32>
  return %Delta : tensor<1x1x128xf32>
}
