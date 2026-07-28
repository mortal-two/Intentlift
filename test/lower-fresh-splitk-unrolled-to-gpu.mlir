// RUN: intent-opt -convert-intent-splitk-reduce-varargs-to-gpu %s | FileCheck %s --check-prefix=DEFAULT
// RUN: intent-opt -convert-intent-splitk-reduce-varargs-to-gpu='rows-per-block=8' %s | FileCheck %s --check-prefix=TUNED

// Exact typed topology materialized from the Fresh14 two-vararg SplitK J.  The
// pass must recognize the weighted-merge algebra from SSA, consume the complete
// graph, and keep the runtime ABI to the four inputs plus two output buffers.
module {
  func.func @fresh_splitk(
      %o0: tensor<1x1x1x2x16xf32>,
      %o1: tensor<1x1x1x2x16xf32>,
      %l0: tensor<1x1x1x2xf32>,
      %l1: tensor<1x1x1x2xf32>)
      -> (tensor<1x1x1x2x16xf32>, tensor<1x1x1x2xf32>) {
    %ninf = arith.constant dense<0xFF800000> : tensor<f32>
    %m0 = intent.map_binary %ninf, %l0 {fn = "max"} :
      (tensor<f32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %m = intent.map_binary %m0, %l1 {fn = "max"} :
      (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %s0 = intent.map_binary %l0, %m {fn = "sub"} :
      (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %log2e = arith.constant dense<1.44269504> : tensor<f32>
    %b0 = intent.map_binary %s0, %log2e {fn = "mul"} :
      (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %w0 = intent.map_unary %b0 {fn = "exp2"} :
      (tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %s1 = intent.map_binary %l1, %m {fn = "sub"} :
      (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %b1 = intent.map_binary %s1, %log2e {fn = "mul"} :
      (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %w1 = intent.map_unary %b1 {fn = "exp2"} :
      (tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %den = intent.map_binary %w0, %w1 {fn = "add"} :
      (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %p0 = intent.contraction %o0, %w0 {subscripts = "abcde,abcd->abcde"} :
      (tensor<1x1x1x2x16xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2x16xf32>
    %p1 = intent.contraction %o1, %w1 {subscripts = "abcde,abcd->abcde"} :
      (tensor<1x1x1x2x16xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2x16xf32>
    %num = intent.map_binary %p0, %p1 {fn = "add"} :
      (tensor<1x1x1x2x16xf32>, tensor<1x1x1x2x16xf32>) -> tensor<1x1x1x2x16xf32>
    %den5 = tensor.expand_shape %den [[0], [1], [2], [3, 4]]
      output_shape [1, 1, 1, 2, 1] :
      tensor<1x1x1x2xf32> into tensor<1x1x1x2x1xf32>
    %q = intent.map_binary %num, %den5 {fn = "div"} :
      (tensor<1x1x1x2x16xf32>, tensor<1x1x1x2x1xf32>) -> tensor<1x1x1x2x16xf32>
    %masked = intent.map_binary %m, %ninf {fn = "eq"} :
      (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xi1>
    %zero = arith.constant dense<0.0> : tensor<f32>
    %masked5 = tensor.expand_shape %masked [[0], [1], [2], [3, 4]]
      output_shape [1, 1, 1, 2, 1] :
      tensor<1x1x1x2xi1> into tensor<1x1x1x2x1xi1>
    %out = intent.select %masked5, %zero, %q :
      (tensor<1x1x1x2x1xi1>, tensor<f32>, tensor<1x1x1x2x16xf32>)
      -> tensor<1x1x1x2x16xf32>
    %lg2 = intent.map_unary %den {fn = "log2"} :
      (tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %ln = intent.map_binary %lg2, %log2e {fn = "div"} :
      (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %lseRaw = intent.map_binary %m, %ln {fn = "add"} :
      (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %lse = intent.select %masked, %m, %lseRaw :
      (tensor<1x1x1x2xi1>, tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>)
      -> tensor<1x1x1x2xf32>
    return %out, %lse : tensor<1x1x1x2x16xf32>, tensor<1x1x1x2xf32>
  }
}

// DEFAULT-LABEL: func.func @fresh_splitk(
// DEFAULT-SAME: %arg0: memref<32xf32>
// DEFAULT-SAME: %arg5: memref<2xf32>
// DEFAULT-SAME: intent.backend.reduction_parts = 2 : i64
// DEFAULT-SAME: intent.backend.reduction_topology = "unrolled_vararg_merge"
// DEFAULT-SAME: intentir.runtime_abi_args = 6 : i64
// DEFAULT: %c128 = arith.constant 128 : index
// DEFAULT: gpu.launch blocks
// DEFAULT: arith.cmpi ult{{.*}}%c16
// DEFAULT: math.log2
// DEFAULT-NOT: intent.

// TUNED-LABEL: func.func @fresh_splitk(
// TUNED-SAME: intent.backend.block_threads = 256 : i64
// TUNED-SAME: intent.backend.rows_per_block = 8 : i64
// TUNED: %c256 = arith.constant 256 : index
// TUNED: gpu.launch blocks
// TUNED-NOT: intent.
