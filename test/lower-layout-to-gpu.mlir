// RUN: intent-opt -convert-intent-layout-to-gpu=block-threads=4 %s | FileCheck %s

// C4 structural layout: intent.concat / pad / tile / flip / triu lowered by
// `convert-intent-layout-to-gpu` to ONE block-cooperative gpu.launch each — one
// block per OUTPUT row, the output column axis split over the block's threads.
// The mechanism is pure DATA MOVEMENT: each output element's SOURCE coord is a
// STATIC structural function of the output coord (affine index / piecewise
// select), then a `memref.load` (or the pad constant). NO gpu.all_reduce, NO
// index-stream load (unlike gather). T=4.

// ---------------------------------------------------------------------------- //
// concat axis=1 (torch.cat / hstack): two inputs placed at cumulative col
// offsets 0 and 4 -> out[i,j] = (j<4) ? a[i,j] : b[i,j-4].
// CHECK-LABEL: func.func @cat1(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK:         scf.if
// CHECK:           memref.load
// CHECK:           memref.store
// CHECK:         gpu.terminator
func.func @cat1(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x8xf32> {
  %c = intent.concat %a, %b {axis = 1 : i64}
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

// -----

// pad lo=[1,1] hi=[1,1] value=0: in-bounds select yields input[i-1,j-1] else 0.
// CHECK-LABEL: func.func @pad0(
// CHECK-SAME:    %{{.*}}: memref<4x6xf32>, %{{.*}}: memref<6x8xf32>)
// CHECK:         gpu.launch
// CHECK:         %[[R:.*]] = scf.if %{{.*}} -> (f32) {
// CHECK:           memref.load
// CHECK:           scf.yield
// CHECK:         } else {
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         memref.store %[[R]]
// CHECK:         gpu.terminator
func.func @pad0(%a: tensor<4x6xf32>) -> tensor<6x8xf32> {
  %p = intent.pad %a {lo = [1, 1], hi = [1, 1], value = 0.0 : f64}
      : (tensor<4x6xf32>) -> tensor<6x8xf32>
  return %p : tensor<6x8xf32>
}

// -----

// tile reps=[2,2]: out[i,j] = input[i mod 2, j mod 4] via arith.remui.
// CHECK-LABEL: func.func @tile0(
// CHECK-SAME:    %{{.*}}: memref<2x4xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.remui
// CHECK:         arith.remui
// CHECK:         memref.load
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @tile0(%a: tensor<2x4xf32>) -> tensor<4x8xf32> {
  %t = intent.tile %a {reps = [2, 2]} : (tensor<2x4xf32>) -> tensor<4x8xf32>
  return %t : tensor<4x8xf32>
}

// -----

// flip axes=[1]: out[i,j] = input[i, N-1-j] (affine reverse on the col axis).
// CHECK-LABEL: func.func @flip1(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.subi
// CHECK:         memref.load
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @flip1(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %f = intent.flip %a {axes = [1]} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %f : tensor<4x8xf32>
}

// -----

// triu k=0: out[i,j] = (j >= i) ? input[i,j] : 0. The load is GUARDED by the
// keep predicate (scf.if) so the zeroed lower triangle is never read.
// CHECK-LABEL: func.func @triu0(
// CHECK-SAME:    %{{.*}}: memref<8x8xf32>, %{{.*}}: memref<8x8xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.cmpi sge
// CHECK:         %[[R:.*]] = scf.if %{{.*}} -> (f32) {
// CHECK:           memref.load
// CHECK:           scf.yield
// CHECK:         } else {
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         memref.store %[[R]]
// CHECK:         gpu.terminator
func.func @triu0(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %u = intent.triu %a {k = 0 : i64} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %u : tensor<8x8xf32>
}
