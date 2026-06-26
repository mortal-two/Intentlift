// RUN: intent-opt %s | intent-opt | FileCheck %s

// Round-trip (parse + verify + print) of the M3 dialect surface added for the
// loss/divergence family: `intent.map_binary` with a comparison fn (gt/ge/lt/
// le/eq) producing an i1-typed condition tensor, and `intent.select` consuming
// it. `select` constrains element types only — the [M,N] i1 condition selects
// between two rank-0 f32 scalars (numpy-style broadcast). This is the tvd grad
// shape: grad = where(p > q, +0.5*scale, -0.5*scale).
// CHECK-LABEL: func.func @tvd_grad(
func.func @tvd_grad(%P: tensor<4x8xf32>, %Q: tensor<4x8xf32>,
                    %pos: tensor<f32>, %neg: tensor<f32>) -> tensor<4x8xf32> {
  // CHECK: intent.map_binary %{{.*}}, %{{.*}} {fn = "gt"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  %cond = intent.map_binary %P, %Q {fn = "gt"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  // CHECK: intent.select %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x8xi1>, tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
  %grad = intent.select %cond, %pos, %neg : (tensor<4x8xi1>, tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
  return %grad : tensor<4x8xf32>
}
