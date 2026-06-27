// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// sparsemax forward + backward, both lowered through the row-reduction pass.
//
// fwd: the new block-cooperative inclusive cumsum (`intent.scan`) path. The row
// uses a CONTIGUOUS column-to-thread tiling (col = tx*K + k) so each thread owns
// a contiguous segment; the full-row cumsum is per-thread sequential prefix ->
// thread-0 sequential exclusive scan of the T segment totals through a
// `gpu.dynamic_shared_memory` [T] buffer (+ gpu.barrier) -> add the exclusive
// prefix. The k-count and s-sum reduces reuse that SAME dynamic-shared scratch as
// a manual tree (NOT gpu.all_reduce, whose static .shared would clobber it). Then
// t=(cssv-1)/(iota+1), support=z>t, tau=(s-1)/max(k,1), y=max(x-tau,0). M=2,N=8,T=4.
// CHECK-LABEL: func.func @sparsemax_fwd(
// CHECK-SAME:    %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<2x8xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK-NOT:     gpu.all_reduce
// CHECK:         gpu.barrier
// CHECK:         arith.sitofp
// CHECK:         arith.cmpf ogt
// CHECK:         arith.maxnumf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @sparsemax_fwd(%x: tensor<2x8xf32>, %sx: tensor<2x8xf32>,
                         %zero: tensor<f32>, %one: tensor<f32>) -> tensor<2x8xf32> {
  %cssv = intent.scan %sx {fn = "sum", axis = 1} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %iota = intent.iota : tensor<8xi32>
  %r = intent.map_binary %iota, %one {fn = "add"} : (tensor<8xi32>, tensor<f32>) -> tensor<8xf32>
  %cm1 = intent.map_binary %cssv, %one {fn = "sub"} : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  %t = intent.map_binary %cm1, %r {fn = "div"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  %support = intent.map_binary %sx, %t {fn = "gt"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
  %selone = intent.select %support, %one, %zero : (tensor<2x8xi1>, tensor<f32>, tensor<f32>) -> tensor<2x8xf32>
  %k = intent.reduce %selone {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2xf32>
  %kc = intent.map_binary %k, %one {fn = "max"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %selz = intent.select %support, %sx, %zero : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  %s = intent.reduce %selz {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2xf32>
  %sm1 = intent.map_binary %s, %one {fn = "sub"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %tau = intent.map_binary %sm1, %kc {fn = "div"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %xmt = intent.map_binary %x, %tau {fn = "sub"} : (tensor<2x8xf32>, tensor<2xf32>) -> tensor<2x8xf32>
  %y = intent.map_binary %xmt, %zero {fn = "max"} : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}

// bwd: NO scan — the M1 reduce-then-broadcast + select template. Two per-row sum
// reduces (go_sum over support, cnt = #support via select(supp,1,0) so no cast),
// then gi = select(o>0, go - go_sum/max(cnt,eps), 0) written from the register
// tile. This is the existing single-tile gpu.all_reduce path (no scan, no
// dynamic-shared). M=4,N=8,T=4.
// CHECK-LABEL: func.func @sparsemax_bwd(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.cmpf ogt
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         arith.divf
// CHECK:         arith.select
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @sparsemax_bwd(%o: tensor<4x8xf32>, %go: tensor<4x8xf32>,
                         %zero: tensor<f32>, %one: tensor<f32>, %eps: tensor<f32>)
    -> tensor<4x8xf32> {
  %supp = intent.map_binary %o, %zero {fn = "gt"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xi1>
  %selgo = intent.select %supp, %go, %zero : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %gosum = intent.reduce %selgo {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %selone = intent.select %supp, %one, %zero : (tensor<4x8xi1>, tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
  %cnt = intent.reduce %selone {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %cntc = intent.map_binary %cnt, %eps {fn = "max"} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %ratio = intent.map_binary %gosum, %cntc {fn = "div"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %inner = intent.map_binary %go, %ratio {fn = "sub"} : (tensor<4x8xf32>, tensor<4xf32>) -> tensor<4x8xf32>
  %gi = intent.select %supp, %inner, %zero : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %gi : tensor<4x8xf32>
}
