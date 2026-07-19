#ifndef INTENT_TARGET_PASSES
#define INTENT_TARGET_PASSES

include "mlir/Pass/PassBase.td"

def LowerGpuMathToNVVM : Pass<"lower-gpu-math-to-nvvm", "::mlir::ModuleOp"> {
  let summary = "Rewrite scalar math transcendentals to native NVVM intrinsics.";
  let description = [{
    Backend codegen peephole for the NVIDIA target. Rewrites scalar `f32`
    `math.exp` inside device code to `2^(x * log2(e))` evaluated by the native
    PTX `ex2.approx.ftz.f32` instruction (emitted as the `llvm.nvvm.ex2.approx.
    ftz.f` LLVM intrinsic via `llvm.call_intrinsic`).

    Motivation: the stock GpuToNVVM lowering turns `math.exp` into a `__nv_expf`
    libdevice call. Two failure modes follow on system MLIR 20.1.8 +
    `--gpu-lower-to-nvvm-pipeline`: (1) with no libdevice on the search path
    ptxas reports `Unresolved extern function '__nv_expf'`; (2) with libdevice
    linked, the serializer internalizes + DCEs the kernel entry, yielding a
    cubin with no `.text` (driver load error 200). The `ex2.approx` form needs
    no libdevice and is exact to ~1e-7 here, which is well inside the softmax
    tolerance. Run this after `gpu-kernel-outlining`, before device lowering.

    This is a generic NVIDIA-backend peephole on `math.*`/LLVM ops; it is
    deliberately NOT part of the `intent` dialect library (it never touches
    `intent.*` ops), hence it lives under lib/Target.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
}

def LowerBf16MmaToNVVM : Pass<"lower-bf16-mma-to-nvvm", "::mlir::ModuleOp"> {
  let summary = "Backport BF16 nvgpu.mma.sync lowering for LLVM 20.";
  let description = [{
    Converts BF16 `nvgpu.mma.sync` m16n8k16 operations, together with their
    neighboring NVGPU async-copy and ldmatrix operations, to the NVVM dialect.
    LLVM/MLIR 20 accepts the BF16 NVGPU IR and exposes the matching NVVM
    intrinsic, but its NVGPU-to-NVVM converter omits BF16 PTX type deduction
    and packed `vector<2xbf16>` register unpacking.  This target-only pass
    backports those two upstream capabilities and is a no-op for modules that
    contain no BF16 MMA operation.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::NVVM::NVVMDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // INTENT_TARGET_PASSES
