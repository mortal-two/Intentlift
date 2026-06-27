// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s

// Llama4 RoPE = the SAME rotation as rope, lowered by the SAME pass with NO
// per-kernel branch. The only difference is the partner pairing: instead of the
// half-split (j, j+hd/2) it is the even/odd complex pair (2j, 2j+1). That is just
// a DIFFERENT STRIDE VALUE in the `intent.strides` arg-attr: q_even / q_odd are
// COL-STRIDE-2 views (strided<[hd, 2]>) of the one [M,hd] buffer.
// new_real = qe*fr - qo*(fi*imag_sign) ; new_imag = qe*(fi*imag_sign) + qo*fr,
// where fr/fi (freqs real/imag) are themselves even/odd views (broadcast rows),
// and imag_sign is a scalar (1.0 fwd / -1.0 bwd, no branch). M=8, hd=8, N=4, T=4.

// In-place (intent.inplace = [0,1]): even/odd results alias qe/qo; signature has
// only the 5 inputs (no fresh outputs) and stores back through the qe/qo views.
// CHECK-LABEL: func.func @llama4_rope(
// CHECK-SAME:    %[[QE:.*]]: memref<8x4xf32, strided<[8, 2]>>, %[[QO:.*]]: memref<8x4xf32, strided<[8, 2]>>, %{{.*}}: memref<4x4xf32, strided<[8, 2]>>, %{{.*}}: memref<4x4xf32, strided<[8, 2]>>, %{{.*}}: memref<f32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK:         memref.load %[[QE]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 2]>>
// CHECK:         arith.divui %{{.*}}, %{{.*}} : index
// CHECK:         arith.mulf
// CHECK:         arith.subf
// CHECK:         memref.store %{{.*}}, %[[QE]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 2]>>
// CHECK:         arith.addf
// CHECK:         memref.store %{{.*}}, %[[QO]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 2]>>
// CHECK:         gpu.terminator
func.func @llama4_rope(%qe: tensor<8x4xf32> {intent.strides = array<i64: 8, 2>},
                       %qo: tensor<8x4xf32> {intent.strides = array<i64: 8, 2>},
                       %fr: tensor<4x4xf32> {intent.strides = array<i64: 8, 2>},
                       %fi: tensor<4x4xf32> {intent.strides = array<i64: 8, 2>},
                       %isign: tensor<f32>)
    -> (tensor<8x4xf32>, tensor<8x4xf32>)
    attributes {intent.inplace = array<i64: 0, 1>} {
  %fis = intent.map_binary %fi, %isign {fn = "mul"} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %a = intent.map_binary %qe, %fr {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %b = intent.map_binary %qo, %fis {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %yr = intent.map_binary %a, %b {fn = "sub"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  %c = intent.map_binary %qe, %fis {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %d = intent.map_binary %qo, %fr {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %yi = intent.map_binary %c, %d {fn = "add"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  return %yr, %yi : tensor<8x4xf32>, tensor<8x4xf32>
}

// CHECK-NOT:     intent.
