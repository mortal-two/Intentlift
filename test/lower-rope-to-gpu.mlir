// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s

// RoPE (rotary position embedding) lowered by the SAME pure-elementwise pass as
// dyt/swiglu/geglu — NO per-kernel branch. The rotation that couples head pair
// (j, j+hd/2) is modeled as an INDEX REMAP: q_lo / q_hi are two strided
// half-views of one [M,hd] buffer (rowStride = hd = 2N via the
// `intent.strides = [hd,1]` arg-attr), so at matching logical column the kernel
// is pure elementwise:  y_lo = q_lo*cos - q_hi*sin ;  y_hi = q_hi*cos + q_lo*sin.
// cos/sin are BROADCAST rows: one row per token (first dim n_row = M/n_head),
// shared by the token's n_head blocks -> loaded at row `bx / n_head` (an
// arith.divui). The strided half-views + broadcast rows are the only new
// mechanism; the math flows through the existing read-once register tile.
// M=8 (n_row=4, n_head=2), hd=8, N=hd/2=4, T=4.

// In-place: y_lo/y_hi alias q_lo/q_hi (intent.inplace = [0,1]) -> the signature
// has NO extra output memrefs (only the 4 inputs) and the rotated result is
// stored back through the input q half-views (memref == input arg).
// CHECK-LABEL: func.func @rope(
// CHECK-SAME:    %[[QLO:.*]]: memref<8x4xf32, strided<[8, 1]>>, %[[QHI:.*]]: memref<8x4xf32, strided<[8, 1]>>, %{{.*}}: memref<4x4xf32, strided<[8, 1]>>, %{{.*}}: memref<4x4xf32, strided<[8, 1]>>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// q half-views loaded at [bx, col]:
// CHECK:         memref.load %[[QLO]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 1]>>
// cos broadcast: token = bx / n_head (n_head = 2) -> divui then load.
// CHECK:         arith.divui %{{.*}}, %{{.*}} : index
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x4xf32, strided<[8, 1]>>
// CHECK:         arith.mulf
// CHECK:         arith.subf
// in-place write-back through the q_lo input arg memref:
// CHECK:         memref.store %{{.*}}, %[[QLO]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 1]>>
// CHECK:         arith.addf
// CHECK:         memref.store %{{.*}}, %[[QHI]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 1]>>
// CHECK:         gpu.terminator
func.func @rope(%q_lo: tensor<8x4xf32> {intent.strides = array<i64: 8, 1>},
                %q_hi: tensor<8x4xf32> {intent.strides = array<i64: 8, 1>},
                %cos: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
                %sin: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>})
    -> (tensor<8x4xf32>, tensor<8x4xf32>)
    attributes {intent.inplace = array<i64: 0, 1>} {
  %a = intent.map_binary %q_lo, %cos {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %b = intent.map_binary %q_hi, %sin {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %ylo = intent.map_binary %a, %b {fn = "sub"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  %c = intent.map_binary %q_hi, %cos {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %d = intent.map_binary %q_lo, %sin {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %yhi = intent.map_binary %c, %d {fn = "add"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  return %ylo, %yhi : tensor<8x4xf32>, tensor<8x4xf32>
}

// CHECK-NOT:     intent.
