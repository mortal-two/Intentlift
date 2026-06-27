// RUN: intent-opt -convert-intent-splitk-reduce-to-gpu %s | FileCheck %s

// The xformers _splitK_reduce / _splitK_reduce_varargs merge: an online
// log-sum-exp combine of the split-K partial outputs, with NO matmul. Expressed
// as ONE intent.loop scanning the split axis of a packed partial output
// O[S,Mq,D] and partial LSE[S,Mq,1] (the varargs list is normalized to this
// packed form upstream, so both reduce variants lift to the SAME IntentIR). The
// loop carries (lse_max[Mq,1], sumexp[Mq,1], O_acc[Mq,D]); the body is a pure
// elementwise/broadcast online update (max/sub/exp/mul/add) -- no contraction,
// no reduce. It lowers to ONE block-cooperative gpu.launch: grid = (Mq,1,1)
// (one block per output row), block = (D,1,1), the split axis scanned with an
// scf.for carrying f32 iter_args. No shared memory, no barrier. No intent.*
// survives.
//
// CHECK-LABEL: func.func @splitk_reduce(
// CHECK-SAME:    %{{.*}}: memref<2x128x64xf32>, %{{.*}}: memref<2x128x1xf32>, %{{.*}}: memref<128x64xf32>)
// CHECK:         gpu.launch
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK:             math.exp
// CHECK:             math.exp
// CHECK:           arith.divf
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.loop
// CHECK-NOT:     intent.contraction
// CHECK-NOT:     intent.reduce
func.func @splitk_reduce(%Osk: tensor<2x128x64xf32>, %Lsk: tensor<2x128x1xf32>)
    -> tensor<128x64xf32> {
  %m0 = arith.constant dense<0xFF800000> : tensor<128x1xf32>
  %se0 = arith.constant dense<0.0> : tensor<128x1xf32>
  %oa0 = arith.constant dense<0.0> : tensor<128x64xf32>
  %r:3 = intent.loop scan(%Osk, %Lsk : tensor<2x128x64xf32>, tensor<2x128x1xf32>)
                     init(%m0, %se0, %oa0 : tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32>)
                     {scan_axes = [0, 0], tile_hint = 1 : i64}
                     -> tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32> do {
  ^bb0(%Os: tensor<128x64xf32>, %Ls: tensor<128x1xf32>,
       %mmax: tensor<128x1xf32>, %se: tensor<128x1xf32>, %oa: tensor<128x64xf32>):
    %mn = intent.map_binary %mmax, %Ls {fn = "max"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %cs = intent.map_binary %mmax, %mn {fn = "sub"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %corr = intent.map_unary %cs {fn = "exp"} : (tensor<128x1xf32>) -> tensor<128x1xf32>
    %ws = intent.map_binary %Ls, %mn {fn = "sub"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %w = intent.map_unary %ws {fn = "exp"} : (tensor<128x1xf32>) -> tensor<128x1xf32>
    %se1 = intent.map_binary %se, %corr {fn = "mul"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %senew = intent.map_binary %se1, %w {fn = "add"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %oa1 = intent.map_binary %oa, %corr {fn = "mul"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
    %wO = intent.map_binary %Os, %w {fn = "mul"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
    %oanew = intent.map_binary %oa1, %wO {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    intent.yield %mn, %senew, %oanew : tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32>
  }
  %O = intent.map_binary %r#2, %r#1 {fn = "div"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
  return %O : tensor<128x64xf32>
}
