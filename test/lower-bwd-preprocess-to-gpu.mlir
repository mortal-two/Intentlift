// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// flash _bwd_preprocess_do_o_dot: delta = sum(o * do, axis=1), a per-row [M]
// output with TWO Row inputs (o, do) and NO [M,N] RowOut. This relaxes the
// row-reduction pass: all-PerRowOut funcs are accepted (no RowOut required) and
// N is derived from a Row INPUT (here o : [M,N]) instead of a RowOut result. The
// reduce result is rank-1 [M] (flash's collapsed delta layout). The cooperative
// register-tile read-once + gpu.all_reduce add + thread-0 scalar write into a
// rank-1 memref<Mxf32> are exactly the existing machinery. M=4, N=8, T=4.
// CHECK-LABEL: func.func @bwd_preprocess(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         arith.mulf
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @bwd_preprocess(%O: tensor<4x8xf32>, %DO: tensor<4x8xf32>) -> tensor<4xf32> {
  %odo = intent.map_binary %O, %DO {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %delta = intent.reduce %odo {fn = "sum", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4xf32>
  return %delta : tensor<4xf32>
}
