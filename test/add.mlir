// RUN: intent-opt %s | FileCheck %s
// An elementwise add at the intent dialect level — the M5 "hello world" the
// back-end lowers all the way to an sm_120a cubin (see test_backend_compile.py).
// CHECK-LABEL: func.func @add
// CHECK: intent.map_binary {{.*}}fn = "add"
func.func @add(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %c = intent.map_binary %a, %b {fn = "add"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %c : tensor<8x16xf32>
}
