// RUN: intent-opt %s | FileCheck %s
// Row softmax over axis 1 — the softmax half of attention. Exercises reduce
// (max/sum keepdims), broadcast map_binary (sub/div against [4x1]), and exp.
// CHECK-LABEL: func.func @softmax
// CHECK: intent.reduce {{.*}}fn = "max"
func.func @softmax(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %s = intent.map_binary %x, %m {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %e = intent.map_unary %s {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %l = intent.reduce %e {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %o = intent.map_binary %e, %l {fn = "div"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}
