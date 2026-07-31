// RUN: intent-opt -lower-row-mask-to-loops='zero-fill=true vector-width=4 require-lowering=true' %s | FileCheck %s

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @rowmask(%data: memref<16x64xf32>, %mask: memref<16xi1>,
                     %output: memref<16x64xf32>) {
    %mask_f32 = memref.alloc() : memref<16xf32>
    linalg.generic {
      indexing_maps = [#id1, #id1], iterator_types = ["parallel"]
    } ins(%mask : memref<16xi1>) outs(%mask_f32 : memref<16xf32>) {
    ^bb0(%value: i1, %unused: f32):
      %cast = arith.uitofp %value : i1 to f32
      linalg.yield %cast : f32
    }
    %temporary = memref.alloc() : memref<16x64xf32>
    linalg.generic {
      indexing_maps = [#id2, #row, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%data, %mask_f32 : memref<16x64xf32>, memref<16xf32>)
      outs(%temporary : memref<16x64xf32>) {
    ^bb0(%value: f32, %factor: f32, %unused: f32):
      %product = arith.mulf %value, %factor : f32
      linalg.yield %product : f32
    }
    memref.copy %temporary, %output
      : memref<16x64xf32> to memref<16x64xf32>
    return
  }
}

// CHECK: module attributes {
// CHECK-SAME: intent.rvv.rowmask_lowering_count = 1 : i64
// CHECK-SAME: intent.rvv.rowmask_mask_hoist_count = 1 : i64
// CHECK-SAME: intent.rvv.rowmask_vector_width = 4 : i64
// CHECK-SAME: intent.rvv.rowmask_zero_fill_count = 1 : i64
// CHECK: scf.for %[[ROW:.*]] =
// CHECK: %[[MASK:.*]] = memref.load %{{.*}}[%[[ROW]]] : memref<16xi1>
// CHECK: scf.if %[[MASK]] {
// CHECK: vector.load %{{.*}}[%[[ROW]], %{{.*}}] : memref<16x64xf32>, vector<4xf32>
// CHECK: vector.store
// CHECK: } else {
// CHECK: vector.splat
// CHECK: vector.store
// CHECK-NOT: linalg.generic
