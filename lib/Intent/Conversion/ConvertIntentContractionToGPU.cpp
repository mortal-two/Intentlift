//===- ConvertIntentContractionToGPU.cpp - matmul -> tensor-core gpu.launch ===//
//
// Lowers a func whose body is exactly one `intent.contraction{"mk,nk->mn"}`
// (A[M,K] f16, W[N,K] f16 -> Y[M,N] f32) to ONE tiled-K tensor-core
// `gpu.launch`. This is the first CUDA-core-free path: the contraction is
// realized by NVIDIA WMMA (`gpu.subgroup_mma_*` -> wmma.mma.sync.aligned
// .row.row.m16n16k16.f32.f32 on sm_120a), NOT scalarized to a `linalg.generic`
// MAC (that leg is left in ConvertIntentToLinalg for the CPU/oracle path).
//
// Schedule (BM=BN=64, BK=16; a 4x4 grid of m16n16k16 C tiles, 4 warps/block):
//   grid  = (cdiv(M,BM)*cdiv(N,BN), 1, 1)  flat pid -> (pid_m, pid_n)
//   block = (128, 1, 1)                    warp w = tx/32 owns C rows [w*16,+16)
// Per block:
//   4 fp32 COp accumulator fragments (one per the warp's 4 column tiles), init 0
//   scf.for kt = 0..K step BK iter_args(the 4 COp fragments):
//     cooperatively copy A[BM,BK] and W[BN,BK] sub-tiles global->dynamic-shared
//       (coalesced, BM*BK/128 = 8 elems/thread), gpu.barrier
//     load the warp's A row tile (AOp) from A_sh; for j in 0..3 load the W
//       column tile (BOp, transposed so "nk" -> the k-major B operand) from W_sh
//       and subgroup_mma_compute COp_j += A * B_j; gpu.barrier; yield
//   epilogue: subgroup_mma_store_matrix each COp_j to Y[w*16, j*16]
//
// The transposed B load makes "mk,nk->mn" exact: B[k,n] = W[n,k], so the MMA
// computes C[m,n] += sum_k A[m,k]*W[n,k] (verified rel ~1e-7 vs torch.matmul).
// Static shapes, M%64==N%64==K%16==0 (tail-mask is a documented follow-up).
// Bare-ptr ABI; the lowered module goes to lower_memref_module_to_kernels with
// scf_to_cf=True (the gpu->nvvm pipeline needs scf->cf BEFORE it to legalize an
// scf.for carrying !gpu.mma_matrix iter_args).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONTRACTIONTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Tensor-core tile geometry. BM/BN = the block's C tile (64x64), BK = the MMA
// K-step (16). Four warps each own a 16x64 C strip = 4 m16n16k16 tiles.
static constexpr int64_t BM = 64, BN = 64, BK = 16, kWMMA = 16, kBlock = 128;

/// One contraction func's tensor-core lowering. Reads the original (tensor)
/// func body (exactly one intent.contraction), builds a new memref+gpu.launch
/// func with the same name, erases the old.
struct ContractionLowering {
  func::FuncOp fn;

  // populated by match()
  int64_t M = 0, N = 0, K = 0;
  unsigned aArgIdx = 0, wArgIdx = 0; // which func args are A[M,K] / W[N,K]
  ContractionOp con;
  Type f16, f32;

  explicit ContractionLowering(func::FuncOp fn) : fn(fn) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 1)
      return failure();
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    // Body: exactly one intent.contraction, returned as the sole result.
    ContractionOp c;
    for (Operation &op : body.without_terminator()) {
      if (auto cc = dyn_cast<ContractionOp>(&op)) {
        if (c)
          return failure(); // more than one contraction
        c = cc;
      } else {
        return failure(); // any other op (epilogue) is M7b, not M7a
      }
    }
    if (!c)
      return failure();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != c.getResult())
      return failure();
    if (c.getSubscripts() != "mk,nk->mn")
      return failure();

    // A[M,K] f16, W[N,K] f16, Y[M,N] f32, all rank-2 static. A and W must be the
    // func's block arguments (destination-passing maps them to memref args).
    auto aArg = dyn_cast<BlockArgument>(c.getA());
    auto wArg = dyn_cast<BlockArgument>(c.getB());
    if (!aArg || !wArg || aArg.getOwner() != &body || wArg.getOwner() != &body)
      return failure();
    aArgIdx = aArg.getArgNumber();
    wArgIdx = wArg.getArgNumber();

    auto aTy = dyn_cast<RankedTensorType>(c.getA().getType());
    auto wTy = dyn_cast<RankedTensorType>(c.getB().getType());
    auto yTy = dyn_cast<RankedTensorType>(c.getResult().getType());
    if (!aTy || !wTy || !yTy || !aTy.hasStaticShape() ||
        !wTy.hasStaticShape() || !yTy.hasStaticShape())
      return failure();
    if (aTy.getRank() != 2 || wTy.getRank() != 2 || yTy.getRank() != 2)
      return failure();
    if (!isa<Float16Type>(aTy.getElementType()) ||
        !isa<Float16Type>(wTy.getElementType()) ||
        !isa<Float32Type>(yTy.getElementType()))
      return failure();
    f16 = aTy.getElementType();
    f32 = yTy.getElementType();

    M = aTy.getDimSize(0);
    K = aTy.getDimSize(1);
    N = wTy.getDimSize(0);
    if (wTy.getDimSize(1) != K || yTy.getDimSize(0) != M ||
        yTy.getDimSize(1) != N)
      return failure();
    // Divisibility: full tiles only (non-divisible tail-mask is a follow-up).
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
      return failure();
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: original args as memrefs ++ the destination-passing Y.
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      auto t = cast<RankedTensorType>(at);
      inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    inTys.push_back(MemRefType::get({M, N}, f32));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value aMem = entry->getArgument(aArgIdx);
    Value wMem = entry->getArgument(wArgIdx);
    Value yMem = entry->getArgument(fn.getFunctionType().getNumInputs());

    int64_t gridN = N / BN, gridM = M / BM;
    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM * gridN);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    // Dynamic shared: A staging [BM,BK] f16 ++ W staging [BN,BK] f16.
    int64_t aBytes = BM * BK * 2, wBytes = BN * BK * 2;
    Value shBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr((int32_t)(aBytes + wBytes)));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value cGridN = b.create<arith::ConstantIndexOp>(loc, gridN);
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cBK = b.create<arith::ConstantIndexOp>(loc, BK);
    Value cK = b.create<arith::ConstantIndexOp>(loc, K);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cWMMA = b.create<arith::ConstantIndexOp>(loc, kWMMA);

    // Flat pid -> (pid_m, pid_n) -> this block's C tile origin (rowBlk, colBlk).
    Value pidM = b.create<arith::DivUIOp>(loc, bx, cGridN);
    Value pidN = b.create<arith::RemUIOp>(loc, bx, cGridN);
    Value cBMv = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBNv = b.create<arith::ConstantIndexOp>(loc, BN);
    Value rowBlk = b.create<arith::MulIOp>(loc, pidM, cBMv);
    Value colBlk = b.create<arith::MulIOp>(loc, pidN, cBNv);
    // warp w = tx/32 owns C rows [rowBlk + w*16, +16).
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value warpRow = b.create<arith::MulIOp>(loc, warp, cWMMA);
    Value myRow = b.create<arith::AddIOp>(loc, rowBlk, warpRow);

    // Dynamic-shared staging buffers (the G1-proven config: dynamic_shared +
    // memref.view, not a workgroup memref.alloca).
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto aShTy = MemRefType::get({BM, BK}, f16, nullptr, wgSpace);
    auto wShTy = MemRefType::get({BN, BK}, f16, nullptr, wgSpace);
    Value aSh = b.create<memref::ViewOp>(loc, aShTy, raw, c0, ValueRange{});
    Value wOff = b.create<arith::ConstantIndexOp>(loc, aBytes);
    Value wSh = b.create<memref::ViewOp>(loc, wShTy, raw, wOff, ValueRange{});

    auto copTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f32, "COp");
    auto aopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "AOp");
    auto bopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "BOp");
    IntegerAttr ldA = b.getIndexAttr(BK);   // A_sh leading dim
    IntegerAttr ldW = b.getIndexAttr(BK);   // W_sh leading dim
    IntegerAttr ldY = b.getIndexAttr(N);    // Y leading dim
    UnitAttr transpose = b.getUnitAttr();

    // 4 fp32 accumulator fragments (one per the warp's 4 column tiles), init 0.
    Value zeroF32 = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    SmallVector<Value> accInit;
    for (int j = 0; j < 4; ++j)
      accInit.push_back(
          b.create<gpu::SubgroupMmaConstantMatrixOp>(loc, copTy, zeroF32));

    // Cooperative copy geometry: BM*BK elements over kBlock threads.
    int64_t copyIters = (BM * BK) / kBlock; // == 8 for 64x16 / 128

    auto forOp = b.create<scf::ForOp>(
        loc, c0, cK, cBK, accInit,
        [&](OpBuilder &nb, Location nl, Value kt, ValueRange acc) {
          // Cooperative coalesced copy: A[rowBlk+r, kt+c] -> A_sh[r,c],
          // W[colBlk+r, kt+c] -> W_sh[r,c]. Unrolled (copyIters static).
          for (int64_t it = 0; it < copyIters; ++it) {
            Value base = nb.create<arith::ConstantIndexOp>(nl, it * kBlock);
            Value i = nb.create<arith::AddIOp>(nl, tx, base);
            Value r = nb.create<arith::DivUIOp>(nl, i, cBK);
            Value cc = nb.create<arith::RemUIOp>(nl, i, cBK);
            Value gk = nb.create<arith::AddIOp>(nl, kt, cc);
            Value ar = nb.create<arith::AddIOp>(nl, rowBlk, r);
            Value av = nb.create<memref::LoadOp>(nl, aMem, ValueRange{ar, gk});
            nb.create<memref::StoreOp>(nl, av, aSh, ValueRange{r, cc});
            Value wr = nb.create<arith::AddIOp>(nl, colBlk, r);
            Value wv = nb.create<memref::LoadOp>(nl, wMem, ValueRange{wr, gk});
            nb.create<memref::StoreOp>(nl, wv, wSh, ValueRange{r, cc});
          }
          nb.create<gpu::BarrierOp>(nl);

          // Load this warp's A row tile once; for each of the 4 column tiles load
          // the (transposed) W tile and accumulate COp_j += A * B_j.
          Value aFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
              nl, aopTy, aSh, ValueRange{warpRow, c0}, ldA, /*transpose=*/nullptr);
          SmallVector<Value> next;
          for (int j = 0; j < 4; ++j) {
            Value cj = nb.create<arith::ConstantIndexOp>(nl, j * kWMMA);
            Value bFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
                nl, bopTy, wSh, ValueRange{cj, c0}, ldW, transpose);
            next.push_back(nb.create<gpu::SubgroupMmaComputeOp>(
                nl, copTy, aFrag, bFrag, acc[j], /*a_transpose=*/nullptr,
                /*b_transpose=*/nullptr));
          }
          nb.create<gpu::BarrierOp>(nl); // before next trip overwrites staging
          nb.create<scf::YieldOp>(nl, next);
        });

    // Epilogue: store each accumulator to Y[myRow, colBlk + j*16].
    for (int j = 0; j < 4; ++j) {
      Value cj = b.create<arith::ConstantIndexOp>(loc, j * kWMMA);
      Value yCol = b.create<arith::AddIOp>(loc, colBlk, cj);
      b.create<gpu::SubgroupMmaStoreMatrixOp>(loc, forOp.getResult(j), yMem,
                                              ValueRange{myRow, yCol}, ldY,
                                              /*transpose=*/nullptr);
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentContractionToGPUPass
    : ::intent::impl::ConvertIntentContractionToGPUBase<
          ConvertIntentContractionToGPUPass> {
  using ConvertIntentContractionToGPUBase::
      ConvertIntentContractionToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ContractionLowering probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ContractionLowering low(fn);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("contraction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
