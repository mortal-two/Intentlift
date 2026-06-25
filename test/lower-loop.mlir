// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// intent.loop (the FlashAttention K-loop: scan K/V tiles, carry the online-
// softmax state m/l/Oacc) lowers to an scf.for over the trip count with the
// carries as tensor iter_args and each tile sliced by tensor.extract_slice. The
// body's intent ops lower to linalg.generic. No intent.* survives.
// CHECK-LABEL: func.func @flash_loop
// CHECK: scf.for %{{.*}} = %c0 to %c4 step %c1 iter_args
// CHECK-SAME: -> (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>)
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [2, 4] [1, 1]
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK-NOT: intent.loop
// CHECK-NOT: intent.yield
func.func @flash_loop(%Qs: tensor<4x4xf32>, %K: tensor<8x4xf32>, %V: tensor<8x4xf32>,
                      %m0: tensor<4x1xf32>, %l0: tensor<4x1xf32>, %O0: tensor<4x4xf32>)
    -> (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>) {
  %r:3 = intent.loop scan(%K, %V : tensor<8x4xf32>, tensor<8x4xf32>)
                     init(%m0, %l0, %O0 : tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>)
                     {scan_axes = [0, 0], tile_hint = 2 : i64}
                     -> tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32> do {
  ^bb0(%Kj: tensor<2x4xf32>, %Vj: tensor<2x4xf32>,
       %m: tensor<4x1xf32>, %l: tensor<4x1xf32>, %Oacc: tensor<4x4xf32>):
    %S = intent.contraction %Qs, %Kj {subscripts = "qd,kd->qk"} : (tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<4x2xf32>
    %mb = intent.reduce %S {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %mn = intent.map_binary %m, %mb {fn = "max"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %Sc = intent.map_binary %S, %mn {fn = "sub"} : (tensor<4x2xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %P = intent.map_unary %Sc {fn = "exp"} : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %co = intent.map_binary %m, %mn {fn = "sub"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %al = intent.map_unary %co {fn = "exp"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %ps = intent.reduce %P {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %ls = intent.map_binary %l, %al {fn = "mul"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %ln = intent.map_binary %ls, %ps {fn = "add"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %PV = intent.contraction %P, %Vj {subscripts = "qk,kd->qd"} : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %Os = intent.map_binary %Oacc, %al {fn = "mul"} : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
    %On = intent.map_binary %Os, %PV {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    intent.yield %mn, %ln, %On : tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>
  }
  return %r#0, %r#1, %r#2 : tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>
}
