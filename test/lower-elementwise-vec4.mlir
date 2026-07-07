// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=32 %s | FileCheck %s

// N3 float4 path: a contiguous pure-elementwise map with K=N/T divisible by 4
// (N=256, T=32 -> K=8 -> 2 vec4 tiles) retypes every Row arg + result to a
// memref<Mx(N/4)xvector<4xf32>> so the bulk load/store lower to ld/st.global.v4.
// silu(x) = x * sigmoid(x): one Row input, one Row output, no iota/select.
// CHECK-LABEL: func.func @silu_vec4(
// CHECK-SAME:    %{{.*}}: memref<4x64xvector<4xf32>>, %{{.*}}: memref<4x64xvector<4xf32>>)
// CHECK:         gpu.launch
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x64xvector<4xf32>>
// CHECK:         vector.extract
// CHECK:         math.exp
// CHECK:         vector.from_elements %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x64xvector<4xf32>>
// CHECK:         gpu.terminator
func.func @silu_vec4(%x: tensor<4x256xf32>) -> tensor<4x256xf32> {
  %s = intent.map_unary %x {fn = "sigmoid"} : (tensor<4x256xf32>) -> tensor<4x256xf32>
  %y = intent.map_binary %x, %s {fn = "mul"} : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  return %y : tensor<4x256xf32>
}

// A strided-view output (result_strides != contiguous) stays on the SCALAR path
// (a half-view cannot be a contiguous vec4 store) — no vector<4xf32> retype.
// CHECK-LABEL: func.func @strided_scalar(
// CHECK-NOT:     vector<4xf32>
// CHECK:         gpu.terminator
func.func @strided_scalar(%x: tensor<4x256xf32>) -> tensor<4x256xf32>
    attributes {intent.result_strides = array<i64: 512, 1>} {
  %y = intent.map_unary %x {fn = "tanh"} : (tensor<4x256xf32>) -> tensor<4x256xf32>
  return %y : tensor<4x256xf32>
}
