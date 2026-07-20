// RUN: intent-opt -convert-intent-sort-to-gpu=block-threads=4 \
// RUN:   -convert-intent-gather-to-gpu=block-threads=2 \
// RUN:   -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// General op-graph dispatch (the keystone): a multi-op program is SEQUENCED into
// outlined single-op / single-chain stage funcs, each of which ONE existing
// cluster / row-reduction pass lowers UNCHANGED, wired by intermediate buffers at
// the driver (see python/intentir/backend/cuda/opgraph.py). This test drives all three
// passes in one intent-opt pipeline over the outlined quantile op-graph
// (sort C6b -> gather C3 -> lerp reduce) plus the topk_softmax op-graph
// (softmax reduce -> topk sort): every stage fires on ITS op within the module,
// proving the passes compose across a multi-func program.

// --- quantile stage 1: sort ascending, keep the first W=hi+1 columns -------- //
// CHECK-LABEL: func.func @q_sort(
// CHECK-SAME:    memref<8x16xf32>, %{{.*}}: memref<8x4xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.barrier
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.sort
func.func @q_sort(%x: tensor<8x16xf32>) -> tensor<8x4xf32> {
  %v, %i = intent.sort %x {descending = false, axis = 1 : i64}
      : (tensor<8x16xf32>) -> (tensor<8x4xf32>, tensor<8x4xi32>)
  return %v : tensor<8x4xf32>
}

// --- quantile stage 2: gather the [lo,hi] columns (data-dependent address) --- //
// CHECK-LABEL: func.func @q_gather(
// CHECK-SAME:    memref<8x4xf32>, %{{.*}}: memref<8x2xi64>, %{{.*}}: memref<8x2xf32>)
// CHECK:         gpu.launch
// CHECK:         %[[RAW:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x2xi64>
// CHECK:         %[[IDX:.*]] = arith.index_cast %[[RAW]] : i64 to index
// CHECK:         memref.load %{{.*}}[%{{.*}}, %[[IDX]]] : memref<8x4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.gather
func.func @q_gather(%data: tensor<8x4xf32>, %index: tensor<8x2xi64>) -> tensor<8x2xf32> {
  %g = intent.gather %data, %index {axis = 1 : i64, mode = "along_axis"}
      : (tensor<8x4xf32>, tensor<8x2xi64>) -> tensor<8x2xf32>
  return %g : tensor<8x2xf32>
}

// --- quantile stage 3: weighted-sum lerp over the two gathered values ------- //
// CHECK-LABEL: func.func @q_lerp(
// CHECK:         gpu.launch
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.reduce
func.func @q_lerp(%g: tensor<8x2xf32>, %w: tensor<8x2xf32>) -> tensor<8x1xf32> {
  %p = intent.map_binary %g, %w {fn = "mul"} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
  %o = intent.reduce %p {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<8x2xf32>) -> tensor<8x1xf32>
  return %o : tensor<8x1xf32>
}

// --- topk_softmax stage 1: row-local safe softmax over the experts ---------- //
// CHECK-LABEL: func.func @tk_softmax(
// CHECK:         gpu.launch
// CHECK:         gpu.all_reduce{{.*}}maxnumf
// CHECK:         math.exp
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.reduce
func.func @tk_softmax(%t: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %m = intent.reduce %t {fn = "max", axes = [1], assoc = "associative"} : (tensor<8x16xf32>) -> tensor<8x1xf32>
  %sm = intent.map_binary %t, %m {fn = "sub"} : (tensor<8x16xf32>, tensor<8x1xf32>) -> tensor<8x16xf32>
  %e = intent.map_unary %sm {fn = "exp"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %d = intent.reduce %e {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<8x16xf32>) -> tensor<8x1xf32>
  %o = intent.map_binary %e, %d {fn = "div"} : (tensor<8x16xf32>, tensor<8x1xf32>) -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}

// --- topk_softmax stage 2: topk = a narrow descending sort (values + idx) --- //
// CHECK-LABEL: func.func @tk_sort(
// CHECK-SAME:    memref<8x16xf32>, %{{.*}}: memref<8x4xf32>, %{{.*}}: memref<8x4xi32>)
// CHECK:         gpu.launch
// CHECK:         gpu.barrier
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.sort
func.func @tk_sort(%x: tensor<8x16xf32>) -> (tensor<8x4xf32>, tensor<8x4xi32>) {
  %v, %i = intent.sort %x {descending = true, axis = 1 : i64}
      : (tensor<8x16xf32>) -> (tensor<8x4xf32>, tensor<8x4xi32>)
  return %v, %i : tensor<8x4xf32>, tensor<8x4xi32>
}
