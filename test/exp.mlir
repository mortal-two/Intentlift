// RUN: intent-opt %s | FileCheck %s
// Elementwise exp at the intent dialect level — the softmax primitive. The
// back-end lowers it through linalg, then the NVVM peephole rewrites math.exp
// to the native ex2.approx PTX instruction (no libdevice), all the way to an
// sm_120a cubin that executes (see test_backend_execute.py::exp).
// CHECK-LABEL: func.func @expf
// CHECK: intent.map_unary {{.*}}fn = "exp"
func.func @expf(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "exp"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}
