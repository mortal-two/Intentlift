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

// Evidence-only live extraction may preserve sigmoid as the primitive source
// graph 1/(1+exp(0-A)). Tensor splat literals must remain on the same one-launch
// vectorized path without a hand-authored rewrite to intent.map_unary sigmoid.
// CHECK-LABEL: func.func @primitive_swiglu_vec4(
// CHECK-SAME:    %{{.*}}: memref<4x64xvector<4xf32>>, %{{.*}}: memref<4x64xvector<4xf32>>, %{{.*}}: memref<4x64xvector<4xf32>>)
// CHECK:         gpu.launch
// CHECK:         arith.constant 0.000000e+00 : f32
// CHECK:         arith.constant 1.000000e+00 : f32
// CHECK:         arith.subf
// CHECK:         math.exp
// CHECK:         arith.addf
// CHECK:         arith.divf
// CHECK:         vector.from_elements %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x64xvector<4xf32>>
// CHECK:         gpu.terminator
func.func @primitive_swiglu_vec4(%A: tensor<4x256xf32>, %B: tensor<4x256xf32>) -> tensor<4x256xf32> {
  %zero = arith.constant dense<0.0> : tensor<f32>
  %neg = intent.map_binary %zero, %A {fn = "sub"} : (tensor<f32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  %exp = intent.map_unary %neg {fn = "exp"} : (tensor<4x256xf32>) -> tensor<4x256xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %den = intent.map_binary %exp, %one {fn = "add"} : (tensor<4x256xf32>, tensor<f32>) -> tensor<4x256xf32>
  %sigmoid = intent.map_binary %one, %den {fn = "div"} : (tensor<f32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  %silu = intent.map_binary %A, %sigmoid {fn = "mul"} : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  %C = intent.map_binary %silu, %B {fn = "mul"} : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  return %C : tensor<4x256xf32>
}

// N4 float2 path: N=256, T=128 -> K=2. This keeps four warps per block while
// preserving one coalesced v2 load/store per thread, matching the structural
// schedule used by the live paper SwiGLU fast path.
// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=128 %s | FileCheck %s --check-prefix=V2
// V2-LABEL: func.func @silu_vec4(
// V2-SAME: memref<4x128xvector<2xf32>>, %{{.*}}: memref<4x128xvector<2xf32>>)
// V2: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x128xvector<2xf32>>
// V2: vector.from_elements %{{.*}}, %{{.*}} : vector<2xf32>
// V2: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x128xvector<2xf32>>

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
