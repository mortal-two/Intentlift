//===- ConvertIntentUpsampleToGPU.cpp - nearest upsample -> gpu.launch ----===//
//
// The C7 interpolation pass. Lowers a func whose body is exactly ONE
// `intent.upsample{mode="nearest"}` over a rank-4 `[N,C,H,W]` f32 input to a
// single block-cooperative `gpu.launch`: grid = (N*C, OH, 1) (one block per
// output channel-plane row), block = (T,1,1) threads over the OW output columns
// (K = OW/T columns each, unrolled).
//
// Pure DATA MOVEMENT (C4-layout-like): each output element's SOURCE coord is an
// affine `floor` of the output coord —
//   ih = min(floor(oh * IH/OH), IH-1),  iw = min(floor(ow * IW/OW), IW-1)
// computed with the reciprocal scale IH/OH in f32 then truncated to int (matching
// flag_gems' `(oh*IH/OH).to(int32)`), followed by a `memref.load` at (n,c,ih,iw).
// Destination-passing: new signature = (input memref) ++ [out]. Bit-EXACT gather
// vs torch + flag_gems. Static shapes, OW % T == 0.
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
#define GEN_PASS_DEF_CONVERTINTENTUPSAMPLETOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct UpsampleLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  UpsampleOp up;
  RankedTensorType inTy, outTy;
  int64_t N = 0, C = 0, H = 0, W = 0, OH = 0, OW = 0;
  int64_t T = 0, K = 0;

  UpsampleLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

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
    up = dyn_cast_or_null<UpsampleOp>(only);
    if (!up || up.getMode() != "nearest")
      return failure();
    inTy = cast<RankedTensorType>(up.getInput().getType());
    outTy = cast<RankedTensorType>(up.getResult().getType());
    if (inTy.getRank() != 4 || outTy.getRank() != 4 || !inTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !inTy.getElementType().isF32())
      return failure();
    N = inTy.getDimSize(0);
    C = inTy.getDimSize(1);
    H = inTy.getDimSize(2);
    W = inTy.getDimSize(3);
    OH = outTy.getDimSize(2);
    OW = outTy.getDimSize(3);
    T = blockThreads ? (int64_t)blockThreads : autoThreads(OW);
    if (T <= 0 || OW % T != 0)
      return failure();
    K = OW / T;
    return success();
  }

  // src = min( (i32)(o * (IN/OUT)), IN-1 ) as an index.
  Value srcCoord(OpBuilder &b, Location loc, Value oIdx, int64_t IN,
                 int64_t OUT) {
    Type f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Value oF = b.create<arith::UIToFPOp>(
        loc, f32, b.create<arith::IndexCastOp>(loc, i32, oIdx));
    Value rscale = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(f32, (double)IN / (double)OUT));
    Value sF = b.create<arith::MulFOp>(loc, oF, rscale);
    Value sI = b.create<arith::FPToSIOp>(loc, i32, sF);  // trunc toward 0 (>=0)
    Value cap = b.create<arith::ConstantIntOp>(loc, IN - 1, 32);
    sI = b.create<arith::MinSIOp>(loc, sI, cap);
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), sI);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    SmallVector<Type> inTys{memrefOf(inTy), memrefOf(outTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    Value outMem = entry->getArgument(1);

    Value cNC = b.create<arith::ConstantIndexOp>(loc, N * C);
    Value cOH = b.create<arith::ConstantIndexOp>(loc, OH);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cNC, cOH, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;  // n*C + c
    Value by = launch.getBlockIds().y;  // oh
    Value tx = launch.getThreadIds().x;

    Value cC = b.create<arith::ConstantIndexOp>(loc, C);
    Value nIdx = b.create<arith::DivUIOp>(loc, bx, cC);
    Value cIdx = b.create<arith::RemUIOp>(loc, bx, cC);
    Value ihIdx = srcCoord(b, loc, by, H, OH);

    for (int64_t k = 0; k < K; ++k) {
      Value ow = tx;
      if (k != 0)
        ow = b.create<arith::AddIOp>(loc, tx,
                                     b.create<arith::ConstantIndexOp>(loc, k * T));
      Value iwIdx = srcCoord(b, loc, ow, W, OW);
      Value v = b.create<memref::LoadOp>(loc, inMem,
                                         ValueRange{nIdx, cIdx, ihIdx, iwIdx});
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{nIdx, cIdx, by, ow});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentUpsampleToGPUPass
    : ::intent::impl::ConvertIntentUpsampleToGPUBase<
          ConvertIntentUpsampleToGPUPass> {
  using ConvertIntentUpsampleToGPUBase::ConvertIntentUpsampleToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      UpsampleLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      UpsampleLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("upsample lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
