// RUN: intent-opt -convert-intent-splitk-reduce-varargs-to-gpu %s | FileCheck %s

// This is a compact structural fixture for the live P4 contract.  It keeps the
// exact four-Out/four-LSE inputs and canonicalizes them with concat(axis=1);
// numerical P3/P6 evidence lives in the paper Python tests/artifacts.
module {
  func.func @structural_candidate(
      %o0: tensor<2x1x4x128x64xf32>,
      %o1: tensor<2x1x4x128x64xf32>,
      %o2: tensor<2x1x4x128x64xf32>,
      %o3: tensor<2x1x4x128x64xf32>,
      %l0: tensor<2x1x4x128xf32>,
      %l1: tensor<2x1x4x128xf32>,
      %l2: tensor<2x1x4x128xf32>,
      %l3: tensor<2x1x4x128xf32>)
      -> (tensor<2x128x1x4x64xf32>, tensor<2x1x4x128xf32>) {
    %zero = arith.constant dense<0.0> : tensor<f32>
    %scale = arith.constant dense<1.44269504> : tensor<f32>
    %ones = arith.constant dense<1.0> : tensor<64xf32>
    %op = intent.concat %o0, %o1, %o2, %o3 {axis = 1 : i64} :
      (tensor<2x1x4x128x64xf32>, tensor<2x1x4x128x64xf32>,
       tensor<2x1x4x128x64xf32>, tensor<2x1x4x128x64xf32>)
      -> tensor<2x4x4x128x64xf32>
    %lp = intent.concat %l0, %l1, %l2, %l3 {axis = 1 : i64} :
      (tensor<2x1x4x128xf32>, tensor<2x1x4x128xf32>,
       tensor<2x1x4x128xf32>, tensor<2x1x4x128xf32>)
      -> tensor<2x4x4x128xf32>
    %mx = intent.reduce %lp {fn = "max", axes = [1], assoc = "associative"} :
      (tensor<2x4x4x128xf32>) -> tensor<2x1x4x128xf32>
    %shift = intent.map_binary %lp, %mx {fn = "sub"} :
      (tensor<2x4x4x128xf32>, tensor<2x1x4x128xf32>) -> tensor<2x4x4x128xf32>
    %base2 = intent.map_binary %shift, %scale {fn = "mul"} :
      (tensor<2x4x4x128xf32>, tensor<f32>) -> tensor<2x4x4x128xf32>
    %w = intent.map_unary %base2 {fn = "exp2"} :
      (tensor<2x4x4x128xf32>) -> tensor<2x4x4x128xf32>
    %sum = intent.reduce %w {fn = "sum", axes = [1], assoc = "float_nondeterministic"} :
      (tensor<2x4x4x128xf32>) -> tensor<2x1x4x128xf32>
    %wd = intent.contraction %w, %ones {subscripts = "zshm,d->zshmd"} :
      (tensor<2x4x4x128xf32>, tensor<64xf32>) -> tensor<2x4x4x128x64xf32>
    %weighted = intent.map_binary %op, %wd {fn = "mul"} :
      (tensor<2x4x4x128x64xf32>, tensor<2x4x4x128x64xf32>)
      -> tensor<2x4x4x128x64xf32>
    %num = intent.reduce %weighted {fn = "sum", axes = [1], assoc = "float_nondeterministic"} :
      (tensor<2x4x4x128x64xf32>) -> tensor<2x1x4x128x64xf32>
    %sumd = intent.contraction %sum, %ones {subscripts = "zghm,d->zghmd"} :
      (tensor<2x1x4x128xf32>, tensor<64xf32>) -> tensor<2x1x4x128x64xf32>
    %acc = intent.map_binary %num, %sumd {fn = "div"} :
      (tensor<2x1x4x128x64xf32>, tensor<2x1x4x128x64xf32>)
      -> tensor<2x1x4x128x64xf32>
    %mxd = intent.contraction %mx, %ones {subscripts = "zghm,d->zghmd"} :
      (tensor<2x1x4x128xf32>, tensor<64xf32>) -> tensor<2x1x4x128x64xf32>
    %condd = intent.map_binary %mxd, %mxd {fn = "eq"} :
      (tensor<2x1x4x128x64xf32>, tensor<2x1x4x128x64xf32>)
      -> tensor<2x1x4x128x64xi1>
    %selected = intent.select %condd, %acc, %acc :
      (tensor<2x1x4x128x64xi1>, tensor<2x1x4x128x64xf32>,
       tensor<2x1x4x128x64xf32>) -> tensor<2x1x4x128x64xf32>
    %out = intent.transpose %selected {perm = [0, 3, 1, 2, 4]} :
      (tensor<2x1x4x128x64xf32>) -> tensor<2x128x1x4x64xf32>
    %cond = intent.map_binary %mx, %mx {fn = "eq"} :
      (tensor<2x1x4x128xf32>, tensor<2x1x4x128xf32>) -> tensor<2x1x4x128xi1>
    %lse = intent.select %cond, %mx, %mx :
      (tensor<2x1x4x128xi1>, tensor<2x1x4x128xf32>, tensor<2x1x4x128xf32>)
      -> tensor<2x1x4x128xf32>
    return %out, %lse : tensor<2x128x1x4x64xf32>, tensor<2x1x4x128xf32>
  }
}

// CHECK-LABEL: func.func @structural_candidate(
// CHECK-SAME: %arg0: memref<65536xf32>
// CHECK-SAME: %arg9: memref<1024xf32>
// CHECK-SAME: %arg54: i32
// CHECK-SAME: intentir.fast_path = "cuda.xformers.splitk-varargs.v1"
// CHECK-SAME: intentir.runtime_abi_args = 55 : i64
// CHECK: gpu.launch blocks
// CHECK: math.exp2
// CHECK: memref.store
// CHECK-NOT: intent.
