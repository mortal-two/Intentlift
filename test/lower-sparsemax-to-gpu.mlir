// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// sparsemax forward + backward, both lowered through the row-reduction pass.
//
// fwd: the new block-cooperative inclusive cumsum (`intent.scan`) path. The row
// uses a CONTIGUOUS column-to-thread tiling (col = tx*K + k) so each thread owns
// a contiguous segment; the full-row cumsum is per-thread sequential prefix ->
// For this sub-warp T=4 case, a deterministic shared-memory exclusive prefix
// avoids target-dependent inactive-lane shuffle behavior. T>=32 retains the
// hierarchical warp-shuffle + compact per-warp exchange path.
// The k-count and s-sum reductions use disjoint compact warp-summary slots
// (NOT gpu.all_reduce, whose static .shared would clobber dynamic scratch). Then
// t=(cssv-1)/(iota+1), support=z>t, tau=(s-1)/max(k,1), y=max(x-tau,0). M=2,N=8,T=4.
// CHECK-LABEL: func.func @sparsemax_fwd(
// CHECK-SAME:    %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<2x8xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK-NOT:     gpu.all_reduce
// CHECK:         memref.view {{.*}} to memref<4xf32, #gpu.address_space<workgroup>>
// CHECK:         memref.store {{.*}} : memref<4xf32, #gpu.address_space<workgroup>>
// CHECK:         gpu.barrier
// CHECK:         memref.load {{.*}} : memref<4xf32, #gpu.address_space<workgroup>>
// CHECK:         arith.sitofp
// CHECK:         arith.cmpf ogt
// CHECK:         arith.maxnumf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]{{.*}} : memref<2x8xf32>
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

// The unchanged shape of a fresh live-provider extraction: literals are tensor
// arith.constant splats, rank is built in i32 and explicitly cast, and the
// support count is an i1->i32 cast followed by an integer keepdims reduction.
// This must hit by graph structure alone (the symbol deliberately differs from
// the historical sparsemax_fwd test above).
// CHECK-LABEL: func.func @live_provider_scan_graph(
// CHECK-SAME:    %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<2x8xf32>, %{{.*}}: memref<2x8xf32>)
// CHECK:         gpu.launch
// CHECK:         memref.view {{.*}} to memref<4xi32, #gpu.address_space<workgroup>>
// CHECK:         arith.sitofp {{.*}} : i32 to f32
// CHECK:         arith.uitofp {{.*}} : i1 to f32
// CHECK:         arith.fptosi {{.*}} : f32 to i32
// CHECK:         arith.sitofp {{.*}} : i32 to f32
// CHECK:         arith.maxnumf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]{{.*}} : memref<2x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_provider_scan_graph(%x: tensor<2x8xf32>, %sx: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %cssv = intent.scan %sx {fn = "sum", axis = 1} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %iota = intent.iota : tensor<8xi32>
  %one_i32 = arith.constant dense<1> : tensor<i32>
  %rank_i32 = intent.map_binary %iota, %one_i32 {fn = "add"} : (tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
  %rank = intent.cast %rank_i32 {to_dtype = "f32"} : (tensor<8xi32>) -> tensor<8xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %cm1 = intent.map_binary %cssv, %one {fn = "sub"} : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  %t = intent.map_binary %cm1, %rank {fn = "div"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  %support = intent.map_binary %sx, %t {fn = "gt"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
  // Canonical J may spell the closed logical alias as `and`; it has the same
  // typed i1 semantics as `logical_and` throughout dialect and target lowering.
  %guarded_support = intent.map_binary %support, %support {fn = "and"} : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
  %support_i32 = intent.cast %guarded_support {to_dtype = "i32"} : (tensor<2x8xi1>) -> tensor<2x8xi32>
  %k_i32 = intent.reduce %support_i32 {fn = "sum", axes = [1], assoc = "associative"} : (tensor<2x8xi32>) -> tensor<2x1xi32>
  %kc_i32 = intent.map_binary %k_i32, %one_i32 {fn = "max"} : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
  %kc = intent.cast %kc_i32 {to_dtype = "f32"} : (tensor<2x1xi32>) -> tensor<2x1xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %supported = intent.select %guarded_support, %sx, %zero : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  %s = intent.reduce %supported {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2x1xf32>
  %sm1 = intent.map_binary %s, %one {fn = "sub"} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %tau = intent.map_binary %sm1, %kc {fn = "div"} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %xmt = intent.map_binary %x, %tau {fn = "sub"} : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
  %y = intent.map_binary %xmt, %zero {fn = "max"} : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}

// bwd: NO scan — the M1 reduce-then-broadcast + select template. Two per-row sum
// reduces (go_sum over support, cnt = #support via select(supp,1,0) so no cast),
// then gi = select(o>0, go - go_sum/max(cnt,eps), 0) written from the register
// tile. This is the existing single-tile grouped-reduction path (no scan, no
// dynamic-shared). M=4,N=8,T=4.
// CHECK-LABEL: func.func @sparsemax_bwd(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.cmpf ogt
// CHECK:         gpu.shuffle xor
// CHECK:         arith.divf
// CHECK:         arith.select
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]{{.*}} : memref<4x8xf32>
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
