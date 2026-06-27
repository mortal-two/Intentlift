// RUN: intent-opt -convert-intent-splitk-reduce-backward-to-gpu %s | FileCheck %s

// The xformers _splitK_reduce_varargs_backward: the GRADIENT of the online
// log-sum-exp merge. Given the packed split-K partials (Out_splitK[S,Mq,D],
// LSE_splitK[S,Mq,1]) and the merged-output gradients (Out, LSE, DOut, DLSE),
// it redistributes back to each split chunk's gradient dOut_splitK[S,Mq,D],
// dLSE_splitK[S,Mq,1]. The DAG: reduce{max,axis=0} (lse_max), reduce{sum,axis=2}
// (the corr head-dim sum), and broadcast map ops -- no loop, no contraction. It
// lowers to ONE block-cooperative gpu.launch: grid = (Mq,1,1), block = (D,1,1),
// two scf.for over the split axis, the corr head-dim sum a gpu.all_reduce. The
// pass appends dOut_splitK / dLSE_splitK destinations. No intent.* survives.
//
// CHECK-LABEL: func.func @splitk_reduce_bwd(
// CHECK-SAME:    %{{.*}}: memref<2x128x64xf32>, %{{.*}}: memref<2x128x1xf32>, %{{.*}}: memref<128x64xf32>, %{{.*}}: memref<128x1xf32>, %{{.*}}: memref<128x64xf32>, %{{.*}}: memref<128x1xf32>, %{{.*}}: memref<2x128x64xf32>, %{{.*}}: memref<2x128x1xf32>)
// CHECK:         gpu.launch
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK:           math.exp
// CHECK:           scf.for
// CHECK:             math.exp
// CHECK:             gpu.all_reduce
// CHECK:             scf.if
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.reduce
// CHECK-NOT:     intent.map_binary
func.func @splitk_reduce_bwd(
    %Osk: tensor<2x128x64xf32>, %Lsk: tensor<2x128x1xf32>,
    %Out: tensor<128x64xf32>, %Lse: tensor<128x1xf32>,
    %dOut: tensor<128x64xf32>, %dLse: tensor<128x1xf32>)
    -> (tensor<2x128x64xf32>, tensor<2x128x1xf32>) {
  // lse_max = max over the split axis.
  %lmax = intent.reduce %Lsk {fn = "max", axes = [0], assoc = "associative"} : (tensor<2x128x1xf32>) -> tensor<1x128x1xf32>
  // w = exp(LSE_splitK - lse_max) / exp(LSE - lse_max).
  %a = intent.map_binary %Lsk, %lmax {fn = "sub"} : (tensor<2x128x1xf32>, tensor<1x128x1xf32>) -> tensor<2x128x1xf32>
  %ea = intent.map_unary %a {fn = "exp"} : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
  %b = intent.map_binary %Lse, %lmax {fn = "sub"} : (tensor<128x1xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %eb = intent.map_unary %b {fn = "exp"} : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %w = intent.map_binary %ea, %eb {fn = "div"} : (tensor<2x128x1xf32>, tensor<1x128x1xf32>) -> tensor<2x128x1xf32>
  // dOut_splitK = w * DOut.
  %doutsk = intent.map_binary %w, %dOut {fn = "mul"} : (tensor<2x128x1xf32>, tensor<128x64xf32>) -> tensor<2x128x64xf32>
  // corr = sum_d ((Out_splitK - Out) * dOut_splitK).
  %diff = intent.map_binary %Osk, %Out {fn = "sub"} : (tensor<2x128x64xf32>, tensor<128x64xf32>) -> tensor<2x128x64xf32>
  %t = intent.map_binary %diff, %doutsk {fn = "mul"} : (tensor<2x128x64xf32>, tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
  %corr = intent.reduce %t {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x128x64xf32>) -> tensor<2x128x1xf32>
  // dLSE_splitK = w * DLSE + corr.
  %wdl = intent.map_binary %w, %dLse {fn = "mul"} : (tensor<2x128x1xf32>, tensor<128x1xf32>) -> tensor<2x128x1xf32>
  %dlsesk = intent.map_binary %wdl, %corr {fn = "add"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
  return %doutsk, %dlsesk : tensor<2x128x64xf32>, tensor<2x128x1xf32>
}
