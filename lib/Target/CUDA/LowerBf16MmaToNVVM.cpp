//===- LowerBf16MmaToNVVM.cpp - BF16 mma.sync backport -------------------===//
//
// MLIR 20 accepts nvgpu.mma.sync m16n8k16 with BF16 fragments and its NVVM
// dialect exposes llvm.nvvm.mma.m16n8k16.row.col.bf16.  The conversion between
// the two is missing two cases that upstream later added: BF16 PTX type
// deduction and packing each vector<2xbf16> fragment register into one i32.
// This pass supplies a higher-benefit BF16 conversion pattern, then reuses the
// stock NVGPU-to-NVVM patterns for ldmatrix and cp.async.
//
//===----------------------------------------------------------------------===//

#include "Target/CUDA/Passes.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace intent {
#define GEN_PASS_DEF_LOWERBF16MMATONVVM
#include "Target/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static bool containsBf16Mma(ModuleOp module) {
  bool found = false;
  module.walk([&](nvgpu::MmaSyncOp op) {
    auto aType = dyn_cast<VectorType>(op.getMatrixA().getType());
    found |= aType && aType.getElementType().isBF16();
  });
  return found;
}

static SmallVector<Value> unpackBf16Registers(ImplicitLocOpBuilder &builder,
                                              Value operand) {
  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(operand.getType());
  Type pairType = VectorType::get(2, builder.getBF16Type());
  if (!arrayType || arrayType.getElementType() != pairType)
    return {};

  SmallVector<Value> registers;
  registers.reserve(arrayType.getNumElements());
  for (unsigned index = 0; index < arrayType.getNumElements(); ++index) {
    Value pair = builder.create<LLVM::ExtractValueOp>(operand, index);
    registers.push_back(
        builder.create<LLVM::BitcastOp>(builder.getI32Type(), pair));
  }
  return registers;
}

static SmallVector<Value> unpackF32Accumulator(ImplicitLocOpBuilder &builder,
                                               Value operand) {
  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(operand.getType());
  Type pairType = VectorType::get(2, builder.getF32Type());
  if (!arrayType || arrayType.getElementType() != pairType)
    return {};

  SmallVector<Value> scalars;
  scalars.reserve(arrayType.getNumElements() * 2);
  for (unsigned row = 0; row < arrayType.getNumElements(); ++row) {
    Value pair = builder.create<LLVM::ExtractValueOp>(operand, row);
    for (int64_t lane = 0; lane < 2; ++lane) {
      Value index = builder.create<LLVM::ConstantOp>(
          builder.getI64Type(), builder.getI64IntegerAttr(lane));
      scalars.push_back(
          builder.create<LLVM::ExtractElementOp>(pair, index));
    }
  }
  return scalars;
}

static Value packF32Accumulator(ImplicitLocOpBuilder &builder,
                                Type desiredType, Value intrinsicResult) {
  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(desiredType);
  Type pairType = VectorType::get(2, builder.getF32Type());
  if (!arrayType || arrayType.getNumElements() != 2 ||
      arrayType.getElementType() != pairType)
    return {};

  Value result = builder.create<LLVM::UndefOp>(desiredType);
  for (int64_t row = 0; row < 2; ++row) {
    Value pair = builder.create<LLVM::UndefOp>(pairType);
    for (int64_t lane = 0; lane < 2; ++lane) {
      Value scalar = builder.create<LLVM::ExtractValueOp>(
          intrinsicResult, row * 2 + lane);
      Value index = builder.create<LLVM::ConstantOp>(
          builder.getI32Type(), builder.getI32IntegerAttr(lane));
      pair =
          builder.create<LLVM::InsertElementOp>(pairType, pair, scalar, index);
    }
    result = builder.create<LLVM::InsertValueOp>(result, pair, row);
  }
  return result;
}

struct Bf16MmaSyncToNVVM
    : ConvertOpToLLVMPattern<nvgpu::MmaSyncOp> {
  explicit Bf16MmaSyncToNVVM(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<nvgpu::MmaSyncOp>(converter,
                                                 PatternBenefit(10)) {}

  LogicalResult
  matchAndRewrite(nvgpu::MmaSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto aType = dyn_cast<VectorType>(op.getMatrixA().getType());
    auto bType = dyn_cast<VectorType>(op.getMatrixB().getType());
    auto cType = dyn_cast<VectorType>(op.getMatrixC().getType());
    auto resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!aType || !bType || !cType || !resultType ||
        !aType.getElementType().isBF16() ||
        !bType.getElementType().isBF16() ||
        !cType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        aType.getShape() != ArrayRef<int64_t>({4, 2}) ||
        bType.getShape() != ArrayRef<int64_t>({2, 2}) ||
        cType.getShape() != ArrayRef<int64_t>({2, 2}) ||
        op.getMmaShapeAsArray() != std::array<int64_t, 3>{16, 8, 16})
      return rewriter.notifyMatchFailure(
          op, "requires BF16 m16n8k16 fragments with an f32 accumulator");

    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    SmallVector<Value> matrixA =
        unpackBf16Registers(builder, adaptor.getMatrixA());
    SmallVector<Value> matrixB =
        unpackBf16Registers(builder, adaptor.getMatrixB());
    SmallVector<Value> matrixC =
        unpackF32Accumulator(builder, adaptor.getMatrixC());
    if (matrixA.size() != 4 || matrixB.size() != 2 || matrixC.size() != 4)
      return rewriter.notifyMatchFailure(op, "unexpected converted fragments");

    Type intrinsicResultType = LLVM::LLVMStructType::getLiteral(
        getContext(), SmallVector<Type>(4, builder.getF32Type()));
    Value intrinsicResult = builder.create<NVVM::MmaOp>(
        intrinsicResultType, matrixA, matrixB, matrixC,
        /*shape=*/std::array<int64_t, 3>{16, 8, 16},
        /*b1Op=*/std::nullopt,
        /*intOverflow=*/std::nullopt,
        /*multiplicandPtxTypes=*/
        std::array<NVVM::MMATypes, 2>{NVVM::MMATypes::bf16,
                                      NVVM::MMATypes::bf16},
        /*multiplicandLayouts=*/
        std::array<NVVM::MMALayout, 2>{NVVM::MMALayout::row,
                                       NVVM::MMALayout::col});

    Type desiredType = typeConverter->convertType(op.getResult().getType());
    Value packed =
        packF32Accumulator(builder, desiredType, intrinsicResult);
    if (!packed)
      return rewriter.notifyMatchFailure(op, "unexpected result fragment");
    rewriter.replaceOp(op, packed);
    return success();
  }
};

static LogicalResult lowerBf16Mma(ModuleOp module) {
  if (!containsBf16Mma(module))
    return success();

  MLIRContext *context = module.getContext();
  LowerToLLVMOptions options(context);
  LLVMTypeConverter converter(context, options);
  populateGpuMemorySpaceAttributeConversions(
      converter, [](gpu::AddressSpace space) -> unsigned {
        switch (space) {
        case gpu::AddressSpace::Global:
          return static_cast<unsigned>(NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case gpu::AddressSpace::Workgroup:
          return static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case gpu::AddressSpace::Private:
          return 0;
        }
        llvm_unreachable("unsupported GPU memory space");
      });
  converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
    return converter.convertType(IntegerType::get(type.getContext(), 32));
  });

  RewritePatternSet patterns(context);
  patterns.add<Bf16MmaSyncToNVVM>(converter);
  populateNVGPUToNVVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, NVVM::NVVMDialect>();
  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                        target);
  return applyPartialConversion(module, target, std::move(patterns));
}

struct LowerBf16MmaToNVVM
    : ::intent::impl::LowerBf16MmaToNVVMBase<LowerBf16MmaToNVVM> {
  using LowerBf16MmaToNVVMBase::LowerBf16MmaToNVVMBase;

  void runOnOperation() override {
    if (failed(lowerBf16Mma(getOperation())))
      signalPassFailure();
  }
};

} // namespace
