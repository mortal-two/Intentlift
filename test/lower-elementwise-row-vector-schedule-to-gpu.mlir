// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=64 row-vector-width=1' %s | FileCheck %s --check-prefix=SCALAR
// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=64 row-vector-width=4' %s | FileCheck %s --check-prefix=VEC4

// The ordinary one-block-per-row path exposes packet width as a target schedule
// coordinate.  Both runs preserve the same map DAG and launch topology; only
// the contiguous row load/store packet changes.

// SCALAR-LABEL: func.func @activation(
// SCALAR-SAME: memref<4x256xf32>
// SCALAR-SAME: intent.backend.vector_width = 1 : i64
// SCALAR: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x256xf32>
// SCALAR-NOT: vector<4xf32>

// VEC4-LABEL: func.func @activation(
// VEC4-SAME: memref<4x64xvector<4xf32>>
// VEC4-SAME: intent.backend.vector_width = 4 : i64
// VEC4: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x64xvector<4xf32>>
// VEC4: vector.extract
// VEC4: vector.from_elements
// VEC4: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x64xvector<4xf32>>
// VEC4-NOT: intent.

func.func @activation(%A: tensor<4x256xf32>, %B: tensor<4x256xf32>)
    -> tensor<4x256xf32> {
  %sigmoid = intent.map_unary %A {fn = "sigmoid"}
    : (tensor<4x256xf32>) -> tensor<4x256xf32>
  %gate = intent.map_binary %A, %sigmoid {fn = "mul"}
    : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  %out = intent.map_binary %gate, %B {fn = "mul"}
    : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  return %out : tensor<4x256xf32>
}
