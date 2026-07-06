//===- ConvertIntentPolarToGPU.cpp - polar complex build -> gpu.launch ----===//
//
// The C7 complex-dtype pass. Lowers a func whose body is exactly ONE
// `intent.polar` over rank-2 `[M,N]` f32 `abs`/`angle` to a single
// block-cooperative `gpu.launch`: one block per row, the column axis split over T
// threads (K = N/T columns each, unrolled). Destination-passing: two appended
// `[M,N]` output memrefs (`real` then `imag`) are written.
//
// Elementwise, reusing the C0 cos/sin scalar emitters:
//   real = abs*cos(angle),  imag = abs*sin(angle)
// (`math.cos`/`math.sin` -> `cos.approx`/`sin.approx` NVVM hardware ops via the
// `-lower-gpu-math-to-nvvm` peephole). The complex tensor is represented as the
// trailing real/imag PAIR (two real results), compared against
// `torch.view_as_real(torch.polar(...))` (rel < 1e-3). Static shapes, N % T == 0.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTPOLARTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct PolarLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  PolarOp polar;
  RankedTensorType inTy;
  int64_t M = 0, Nc = 0, T = 0, K = 0;

  PolarLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    polar = dyn_cast_or_null<PolarOp>(only);
    if (!polar)
      return failure();
    inTy = cast<RankedTensorType>(polar.getAbs().getType());
    if (inTy.getRank() != 2 || !inTy.hasStaticShape() ||
        !inTy.getElementType().isF32())
      return failure();
    M = inTy.getDimSize(0);
    Nc = inTy.getDimSize(1);
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Nc);
    if (T <= 0 || Nc % T != 0)
      return failure();
    K = Nc / T;
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    MemRefType mem = memrefOf(inTy);
    SmallVector<Type> inTys{mem, mem, mem, mem};  // abs, angle, real, imag
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value absMem = entry->getArgument(0);
    Value angMem = entry->getArgument(1);
    Value reMem = entry->getArgument(2);
    Value imMem = entry->getArgument(3);

    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0)
        col = b.create<arith::AddIOp>(loc, tx,
                                      b.create<arith::ConstantIndexOp>(loc, k * T));
      Value a = b.create<memref::LoadOp>(loc, absMem, ValueRange{bx, col});
      Value ang = b.create<memref::LoadOp>(loc, angMem, ValueRange{bx, col});
      Value re = b.create<arith::MulFOp>(loc, a, b.create<math::CosOp>(loc, ang));
      Value im = b.create<arith::MulFOp>(loc, a, b.create<math::SinOp>(loc, ang));
      b.create<memref::StoreOp>(loc, re, reMem, ValueRange{bx, col});
      b.create<memref::StoreOp>(loc, im, imMem, ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentPolarToGPUPass
    : ::intent::impl::ConvertIntentPolarToGPUBase<ConvertIntentPolarToGPUPass> {
  using ConvertIntentPolarToGPUBase::ConvertIntentPolarToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      PolarLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      PolarLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("polar lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
