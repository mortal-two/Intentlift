// RUN: intent-opt -split-input-file -convert-intent-rng-to-gpu=block-threads=4 %s | FileCheck %s

// C5 stateful RNG: an intent.rand (+ composed elementwise transform) lowered by
// `convert-intent-rng-to-gpu` to ONE block-cooperative gpu.launch — one block
// per OUTPUT row, the column axis split over the block's threads. The mechanism
// is a STATELESS counter-based philox: each element computes
// philox4x32-10(seed, row*Co+col) in arith (10 rounds of mul-hi/xor) -> a
// uniform[0,1) -> the composed transform -> store. NO gpu.all_reduce. T=4.

// rand: uniform[0,1) written straight out (no tensor input; only the appended
// destination memref). The philox counter is row*Co+col; the mul-hi rounds are
// arith.muli over i64 extensions + arith.trunci; the uniform is uitofp(c>>8)*k.
// CHECK-LABEL: func.func @rand(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     gpu.all_reduce
// CHECK:         arith.extui
// CHECK:         arith.muli %{{.*}}, %{{.*}} : i64
// CHECK:         arith.trunci
// CHECK:         arith.uitofp
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @rand() -> tensor<4x8xf32> {
  %r = intent.rand {seed = 12345 : i64} : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

// dropout = select(rand >= p, x/(1-p), 0): the RNG composes the C0/M3 compare +
// select + scale (arith.cmpf oge, arith.mulf, arith.select). The tensor input x
// is passed as a memref; the destination memref is appended.
// CHECK-LABEL: func.func @dropout(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         memref.load
// CHECK:         arith.cmpf oge
// CHECK:         arith.mulf
// CHECK:         arith.select
// CHECK:         memref.store
func.func @dropout(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = intent.rand {seed = 7 : i64} : tensor<4x8xf32>
  %p = arith.constant dense<0.25> : tensor<4x8xf32>
  %keep = intent.map_binary %r, %p {fn = "ge"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  %inv = arith.constant dense<1.3333334> : tensor<4x8xf32>
  %scaled = intent.map_binary %x, %inv {fn = "mul"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %out = intent.select %keep, %scaled, %zero
      : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// -----

// randn (Box-Muller): z = sqrt(-2 ln u1) * cos(2pi u2), two rand streams with
// different seeds. The cos/log/sqrt route through math ops (-> approx hardware
// intrinsics by the target peephole).
// CHECK-LABEL: func.func @randn(
// CHECK:         gpu.launch
// CHECK:         math.log
// CHECK:         math.sqrt
// CHECK:         math.cos
// CHECK:         memref.store
func.func @randn() -> tensor<4x8xf32> {
  %u1 = intent.rand {seed = 1 : i64} : tensor<4x8xf32>
  %u2 = intent.rand {seed = 2 : i64} : tensor<4x8xf32>
  %lnu = intent.map_unary %u1 {fn = "log"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %m2 = arith.constant dense<-2.0> : tensor<4x8xf32>
  %rad2 = intent.map_binary %lnu, %m2 {fn = "mul"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %rad = intent.map_unary %rad2 {fn = "sqrt"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %tau = arith.constant dense<6.2831855> : tensor<4x8xf32>
  %ang = intent.map_binary %u2, %tau {fn = "mul"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %c = intent.map_unary %ang {fn = "cos"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %z = intent.map_binary %rad, %c {fn = "mul"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %z : tensor<4x8xf32>
}
