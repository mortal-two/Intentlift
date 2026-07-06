// RUN: intent-opt -convert-intent-gather-to-gpu=block-threads=4 %s | FileCheck %s

// C3 data-dependent addressing: intent.gather / intent.scatter lowered by
// `convert-intent-gather-to-gpu` to ONE block-cooperative gpu.launch each — one
// block per output/scatter row, the column axis split over the block's threads.
// The NEW mechanism is a RUNTIME load/store coordinate on `data`: the index
// value is `memref.load`ed from the integer index memref, `arith.index_cast`ed
// to an index, then used as a `data` load/store coordinate. NO gpu.all_reduce,
// NO memref.alloca. M=Ro, N=Co, T=4.

// ---------------------------------------------------------------------------- //
// gather along_axis, axis=1 (torch.gather): out[i,j] = data[i, index[i,j]].
// CHECK-LABEL: func.func @gather_along(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xi64>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK:         %[[RAW:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xi64>
// CHECK:         %[[IDX:.*]] = arith.index_cast %[[RAW]] : i64 to index
// CHECK:         memref.load %{{.*}}[%{{.*}}, %[[IDX]]] : memref<4x8xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
func.func @gather_along(%data: tensor<4x8xf32>, %index: tensor<4x8xi64>) -> tensor<4x8xf32> {
  %g = intent.gather %data, %index {axis = 1 : i64, mode = "along_axis"}
      : (tensor<4x8xf32>, tensor<4x8xi64>) -> tensor<4x8xf32>
  return %g : tensor<4x8xf32>
}

// -----

// gather take, axis=0 (embedding / index_select rows): index is rank-1 [Ro];
// out[i,j] = data[index[i], j].
// CHECK-LABEL: func.func @gather_rows(
// CHECK-SAME:    %{{.*}}: memref<6x8xf32>, %{{.*}}: memref<4xi64>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         %[[RAW:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xi64>
// CHECK:         %[[IDX:.*]] = arith.index_cast %[[RAW]] : i64 to index
// CHECK:         memref.load %{{.*}}[%[[IDX]], %{{.*}}] : memref<6x8xf32>
// CHECK:         gpu.terminator
func.func @gather_rows(%data: tensor<6x8xf32>, %index: tensor<4xi64>) -> tensor<4x8xf32> {
  %g = intent.gather %data, %index {axis = 0 : i64, mode = "take"}
      : (tensor<6x8xf32>, tensor<4xi64>) -> tensor<4x8xf32>
  return %g : tensor<4x8xf32>
}

// -----

// scatter along_axis, axis=1, reduce=none: in-place on `data` (no appended out
// arg), a data-dependent memref.store at data[i, index[i,j]] = src[i,j].
// CHECK-LABEL: func.func @scatter_set(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xi64>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         %[[IDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %[[IDX]]] : memref<4x8xf32>
// CHECK-NOT:     memref.atomic_rmw
// CHECK:         gpu.terminator
func.func @scatter_set(%data: tensor<4x8xf32>, %index: tensor<4x8xi64>, %src: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %o = intent.scatter %data, %index, %src {axis = 1 : i64, mode = "along_axis", reduce = "none"}
      : (tensor<4x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}

// -----

// scatter_add: reduce=add -> a data-dependent memref.atomic_rmw addf
// (red.global.add.f32 on sm_120a).
// CHECK-LABEL: func.func @scatter_add(
// CHECK:         gpu.launch
// CHECK:         %[[IDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         memref.atomic_rmw addf %{{.*}}, %{{.*}}[%{{.*}}, %[[IDX]]] : (f32, memref<4x8xf32>) -> f32
// CHECK:         gpu.terminator
func.func @scatter_add(%data: tensor<4x8xf32>, %index: tensor<4x8xi64>, %src: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %o = intent.scatter %data, %index, %src {axis = 1 : i64, mode = "along_axis", reduce = "add"}
      : (tensor<4x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}
