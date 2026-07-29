// RUN: not intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true require-vectorization=true' %s 2>&1 | FileCheck %s

// CHECK: error: required explicit row-reduction vectorization matched no operation

// An ordered reduction must remain outside the observation-changing vector
// tree even when another caller enables the global reassociation conversion
// option.  Admission comes from the reduction's own assoc contract.
func.func @ordered_sum(%x: tensor<4x10xf32>) -> tensor<4xf32> {
  %sum = intent.reduce %x {
    fn = "sum", axes = [1], assoc = "associative"
  } : (tensor<4x10xf32>) -> tensor<4xf32>
  return %sum : tensor<4xf32>
}
