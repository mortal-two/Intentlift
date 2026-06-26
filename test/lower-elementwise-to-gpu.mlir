// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s

// Pure-elementwise activation kernels (NO reduce) lowered by the sibling pass
// `convert-intent-elementwise-to-gpu` to ONE read-once block-cooperative
// gpu.launch each: one block per row, the feature dim split over the block's
// threads, the register tile loaded ONCE into SSA, per-row scalars hoisted once,
// the output written from the register tile. There is NO gpu.all_reduce
// (nothing is reduced) and NO memref.alloca (register tile, not local memory).
// General over the map DAG — NO per-kernel branch; dyt / swiglu / geglu flow
// through identically. Constants (geglu's 0.5 / 0.044715 / sqrt(2/pi) / 1) are
// rank-0 Scalar args (no intent.const op); intent.select/intent.cast are M3b, so
// geglu/swiglu's source `.cast(b.dtype)` is an fp32 no-op here. M=4, N=8, T=4.

// dyt: y = tanh(alpha*x) * gamma  (alpha Scalar, gamma Col[N], x Row[M,N]).
// CHECK-LABEL: func.func @dyt(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         memref.load %{{.*}}[] : memref<f32>
// CHECK:         arith.mulf
// tanh(alpha*x) is computed via math.exp (-> ex2.approx hardware), not the
// libdevice __nv_tanhf extern (a ptxas dead end).
// CHECK:         math.exp
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
func.func @dyt(%X: tensor<4x8xf32>, %Alpha: tensor<f32>, %Gamma: tensor<8xf32>) -> tensor<4x8xf32> {
  %ax = intent.map_binary %Alpha, %X {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %t = intent.map_unary %ax {fn = "tanh"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %y = intent.map_binary %t, %Gamma {fn = "mul"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// swiglu: c = silu(a) * b = (a * sigmoid(a)) * b  (a, b Row[M,N]).
// CHECK-LABEL: func.func @swiglu(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         math.exp
// CHECK:         arith.mulf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
func.func @swiglu(%A: tensor<4x8xf32>, %B: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %s = intent.map_unary %A {fn = "sigmoid"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %silu = intent.map_binary %A, %s {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %c = intent.map_binary %silu, %B {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

// geglu: c = (0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3)))) * b  (a, b Row).
// CHECK-LABEL: func.func @geglu(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         math.exp
// CHECK:         arith.mulf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
func.func @geglu(%A: tensor<4x8xf32>, %B: tensor<4x8xf32>,
                 %half: tensor<f32>, %k: tensor<f32>, %sp: tensor<f32>, %one: tensor<f32>)
    -> tensor<4x8xf32> {
  %a2 = intent.map_binary %A, %A {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %a3 = intent.map_binary %a2, %A {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ka3 = intent.map_binary %k, %a3 {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %inner = intent.map_binary %A, %ka3 {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %arg = intent.map_binary %sp, %inner {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %th = intent.map_unary %arg {fn = "tanh"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %opt = intent.map_binary %one, %th {fn = "add"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ha = intent.map_binary %half, %A {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %gate = intent.map_binary %ha, %opt {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %c = intent.map_binary %gate, %B {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK-NOT:     intent.
