// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=256 %s | FileCheck %s
// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=256 grid-stride-unroll=4' %s | FileCheck %s --check-prefix=UNROLL
// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=128 logical-tile-elements=4096' %s | FileCheck %s --check-prefix=TILE

// A large static pure-map domain must not materialize N/T copies of the map DAG
// in compiler IR.  The generic elementwise pass reuses the shared flat
// grid-stride emitter: 1,048,576 elements need 4096 blocks at 256 threads, below
// the four-wave occupancy cap.  The loop bound handles the tail for arbitrary N.
//
// CHECK-LABEL: func.func @large_identity(
// CHECK-SAME: memref<1048576xf32>
// CHECK-SAME: intent.backend.grid_blocks = 4096 : i64
// CHECK-SAME: intent.backend.grid_stride_loop_count = 1 : i64
// CHECK-SAME: "cuda.elementwise.flat_grid_stride.v1"
// CHECK: %[[GRID:.*]] = arith.constant 4096 : index
// CHECK: %[[BLOCK:.*]] = arith.constant 256 : index
// CHECK: gpu.launch{{.*}}%[[GRID]]{{.*}}%[[BLOCK]]
// CHECK: scf.for
// CHECK: arith.remui
// CHECK-COUNT-1: memref.load
// CHECK-COUNT-1: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: intent.
// UNROLL-LABEL: func.func @large_identity(
// UNROLL-SAME: intent.backend.grid_blocks = 1024 : i64
// UNROLL-SAME: intent.backend.grid_stride_unroll = 4 : i64
// UNROLL-SAME: "cuda.elementwise.flat_grid_stride.unroll.v1"
// UNROLL: %[[UGRID:.*]] = arith.constant 1024 : index
// UNROLL: gpu.launch{{.*}}%[[UGRID]]
// UNROLL: scf.for
// UNROLL-COUNT-4: memref.load
// UNROLL-COUNT-4: memref.store
// TILE-LABEL: func.func @large_identity(
// TILE-SAME: intent.backend.grid_blocks = 256 : i64
// TILE-SAME: intent.backend.logical_tile_elements = 4096 : i64
// TILE-SAME: "cuda.elementwise.flat_logical_tile.v1"
// TILE: %[[TGRID:.*]] = arith.constant 256 : index
// TILE: %[[TBLOCK:.*]] = arith.constant 128 : index
// TILE: gpu.launch{{.*}}%[[TGRID]]{{.*}}%[[TBLOCK]]
// TILE: scf.for
func.func @large_identity(%x: tensor<1048576xf32>) -> tensor<1048576xf32> {
  %y = intent.map_unary %x {fn = "identity"}
    : (tensor<1048576xf32>) -> tensor<1048576xf32>
  return %y : tensor<1048576xf32>
}

// Grid-stride lowering must derive a broadcast source row from the logical row,
// not from the physical block id.  The latter changes on every grid-stride
// iteration and would read a different row after the first wave.
//
// CHECK-LABEL: func.func @large_row_broadcast(
// CHECK: %[[COL:.*]] = arith.remui %[[FLAT:.*]], %{{.*}} : index
// CHECK: %[[ROW:.*]] = arith.divui %[[FLAT]], %{{.*}} : index
// CHECK: %[[BROADCAST_ROW:.*]] = arith.divui %[[ROW]], %{{.*}} : index
// CHECK: memref.load %{{.*}}[%[[BROADCAST_ROW]], %[[COL]]]
func.func @large_row_broadcast(
    %x: tensor<2x65536xf32>, %row: tensor<1x65536xf32>)
    -> tensor<2x65536xf32> {
  %y = intent.map_binary %x, %row {fn = "add"}
    : (tensor<2x65536xf32>, tensor<1x65536xf32>) -> tensor<2x65536xf32>
  return %y : tensor<2x65536xf32>
}
