// RUN: intent-opt %s -convert-intent-row-reduction-to-gpu="block-threads=64 predicate-pure-value-if=true" | FileCheck %s

// A runtime value_if remains structured in M0.  The C++ row path predicates
// its pure tensor arms, folds the independent constexpr select, and records
// only transformations that actually occurred in the emitted MLIR state.

// CHECK-LABEL: func.func @runtime_value_if
// CHECK-SAME: intent.backend.folded_constexpr_select_count = 1 : i64
// CHECK-SAME: intent.backend.host_value_capture_count = 1 : i64
// CHECK-SAME: intent.backend.predicated_value_if_count = 1 : i64
// CHECK-SAME: intent.backend.transformation_trace = ["cuda.row_reduction.read_once.v1", "cuda.reduction.workgroup_hierarchical.v1", "cuda.control.value_if_to_select.v1", "cuda.control.constexpr_select_fold.v1", "cuda.b.rank0_host_value_capture.v1"]
// CHECK: gpu.launch
// CHECK-NOT: intent.
// CHECK-NOT: tensor.extract
// CHECK-NOT: scf.if %{{.*}} -> (tensor

module {
  func.func @runtime_value_if(
      %x: tensor<2x64xf32>,
      %weight: tensor<64xf32>,
      %predicate: tensor<i1>) -> tensor<2x64xf32> {
    %condition = tensor.extract %predicate[] : tensor<i1>
    %selected = scf.if %condition -> (tensor<64xf32>) {
      %one = arith.constant dense<1.0> : tensor<f32>
      %adjusted = intent.map_binary %weight, %one {fn = "add"} :
          (tensor<64xf32>, tensor<f32>) -> tensor<64xf32>
      scf.yield %adjusted : tensor<64xf32>
    } else {
      scf.yield %weight : tensor<64xf32>
    }
    %false = arith.constant dense<false> : tensor<i1>
    %unused = arith.constant dense<2.0> : tensor<f32>
    %dead = intent.map_binary %weight, %unused {fn = "mul"} :
        (tensor<64xf32>, tensor<f32>) -> tensor<64xf32>
    %folded = intent.select %false, %dead, %selected :
        (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %sum = intent.reduce %x {fn = "sum", axes = [1], assoc = "float_nondeterministic"} :
        (tensor<2x64xf32>) -> tensor<2xf32>
    %scaled = intent.contraction %x, %sum {subscripts = "ab,a->ab"} :
        (tensor<2x64xf32>, tensor<2xf32>) -> tensor<2x64xf32>
    %out = intent.map_binary %scaled, %folded {fn = "mul"} :
        (tensor<2x64xf32>, tensor<64xf32>) -> tensor<2x64xf32>
    return %out : tensor<2x64xf32>
  }
}
