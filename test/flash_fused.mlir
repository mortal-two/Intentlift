// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s
// The fused-flash structure survives the NVVM exp peephole: scf.parallel(query)
// wrapping scf.for+iter_args(carries), with math.exp -> native ex2.approx.
// CHECK-LABEL: func.func @flash
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK-NOT: math.exp
//
// Hand-written FUSED flash forward (probe). One gpu block per query row
// (scf.parallel over Sq -> grid), an inner sequential scf.for over keys with
// SCALAR online-softmax carries (m, l, acc0, acc1) -> iter_args. Body does the
// rescale alpha=exp(m_old-m_new), acc=alpha*acc + p*v_j. D=2, Sq=2, Sk=4.
// math.exp is rewritten to ex2.approx by -lower-gpu-math-to-nvvm.
module {
  func.func @flash(%Q: memref<2x2xf32>, %K: memref<4x2xf32>, %V: memref<4x2xf32>,
                   %O: memref<2x2xf32>, %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    scf.parallel (%qi) = (%c0) to (%c2) step (%c1) {
      %q0 = memref.load %Q[%qi, %c0] : memref<2x2xf32>
      %q1 = memref.load %Q[%qi, %c1] : memref<2x2xf32>
      %res:4 = scf.for %j = %c0 to %c4 step %c1
          iter_args(%m = %ninf, %l = %zero, %a0 = %zero, %a1 = %zero)
          -> (f32, f32, f32, f32) {
        %k0 = memref.load %K[%j, %c0] : memref<4x2xf32>
        %k1 = memref.load %K[%j, %c1] : memref<4x2xf32>
        %p0 = arith.mulf %q0, %k0 : f32
        %p1 = arith.mulf %q1, %k1 : f32
        %dot = arith.addf %p0, %p1 : f32
        %s = arith.mulf %dot, %scale : f32
        %mnew = arith.maximumf %m, %s : f32
        %md = arith.subf %m, %mnew : f32
        %alpha = math.exp %md : f32
        %sd = arith.subf %s, %mnew : f32
        %p = math.exp %sd : f32
        %la = arith.mulf %alpha, %l : f32
        %lnew = arith.addf %la, %p : f32
        %v0 = memref.load %V[%j, %c0] : memref<4x2xf32>
        %v1 = memref.load %V[%j, %c1] : memref<4x2xf32>
        %a0a = arith.mulf %alpha, %a0 : f32
        %pv0 = arith.mulf %p, %v0 : f32
        %a0new = arith.addf %a0a, %pv0 : f32
        %a1a = arith.mulf %alpha, %a1 : f32
        %pv1 = arith.mulf %p, %v1 : f32
        %a1new = arith.addf %a1a, %pv1 : f32
        scf.yield %mnew, %lnew, %a0new, %a1new : f32, f32, f32, f32
      }
      %o0 = arith.divf %res#2, %res#1 : f32
      %o1 = arith.divf %res#3, %res#1 : f32
      memref.store %o0, %O[%qi, %c0] : memref<2x2xf32>
      memref.store %o1, %O[%qi, %c1] : memref<2x2xf32>
      scf.reduce
    }
    return
  }
}
