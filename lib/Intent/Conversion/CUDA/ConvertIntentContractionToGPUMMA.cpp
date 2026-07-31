//===- ConvertIntentContractionToGPUMMA.cpp - matmul -> mma.sync gpu.launch ==//
//
// The nvgpu.ldmatrix + nvgpu.mma.sync sibling of ConvertIntentContractionToGPU
// (the opaque gpu.subgroup_mma_* / !gpu.mma_matrix WMMA path). Same recognizer
// and the same cp.async multistage staging, but the contraction is realized by
// the register-level nvgpu.ldmatrix (shared->register warp-cooperative fragment
// load) + nvgpu.mma.sync (m16n8k16 warp MMA on vector fragments, KNOWN lane
// layout -> vector<2x2xf32> accumulator). The WMMA pass is left fully intact;
// this is a separate pass gated by its own name.
//
// Fragment recipe (validated end-to-end on sm_120a against MLIR's own
// --convert-vector-to-gpu use-nvgpu lowering -- the G4 probe):
//   lane = tid%32, warp = tid/32; each warp owns a 16-row C strip.
//   A operand : ldmatrix aSh[warp*16 + lane%16, kOff + (lane/16)*8]
//               transpose=false -> vector<4x2x(f16|bf16)>
//   W operand : ldmatrix wSh[n*8 + lane%8, kOff + (lane/8)*8]
//               transpose=false -> vector<2x2x(f16|bf16)> (natural [N,K], the
//               "nk" transpose folded into the load -- no transpose-on-copy)
//   mma.sync m16n8k16 -> vector<2x2xf32> accumulator (per warp, one per n8 col
//   tile), carried across the K loop as scf.for iter_args.
//   store     : d[i][j] -> row = warp*16 + lane/4 + i*8,
//                          col = n*8 + (lane%4)*2 + j   (textbook PTX layout)
//
// Static shapes, matching f16/bf16 A/W storage, f32 accumulation, and
// M%BM==N%BN==K%BK==0. Bare-ptr ABI; scf_to_cf=True in
// lower_memref_module_to_kernels (the scf.for carries vector iter_args). LLVM
// 20's missing BF16 NVGPU conversion is supplied by LowerBf16MmaToNVVM.
//
// smem-pad-k: the rank-3 A/W staging buffers pad the K leading dim by `padK`
// 16-bit elements (default 8 = one 128-bit lane) so successive ldmatrix rows
// land on
// distinct shared banks -- the bank-conflict-free analog of wmma_matmul.cuh's
// AS_PAD/BS_PAD. The 128-bit cp.async copies fill only the first BK columns of
// each padded row; the memref stride carries the pad transparently. At the
// compute-bound [4096^3] config (BM=BN=128, ns=2) padding is an ~11% win and
// unlocks the wider 8-warp tile; it + the 2-stage double buffer land the
// mma.sync GEMM at ~1.19x cuBLAS -- the sm_120a wgmma/TMA-less floor.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Schedule.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONTRACTIONTOGPUMMA
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// mma.sync m16n8k16 tile geometry.
static constexpr int64_t MMA_M = 16, MMA_N = 8, MMA_K = 16;

/// Recognize a rank-2 row-by-row contraction independently of the author's
/// choice of einsum letters.  ``mk,nk->mn`` and ``ik,jk->ij`` describe the same
/// implementation space; treating the spelling as a lowering precondition
/// would make otherwise identical J graphs take different backend paths.
static bool isRowByRowContraction(StringRef subscripts) {
  auto arrow = subscripts.split("->");
  if (arrow.second.empty())
    return false;
  auto inputs = arrow.first.split(',');
  StringRef lhs = inputs.first.trim();
  StringRef rhs = inputs.second.trim();
  StringRef out = arrow.second.trim();
  if (lhs.size() != 2 || rhs.size() != 2 || out.size() != 2)
    return false;
  return lhs[0] != lhs[1] && rhs[0] != rhs[1] && lhs[0] != rhs[0] &&
         lhs[1] == rhs[1] && out[0] == lhs[0] && out[1] == rhs[0];
}

struct ContractionMMALowering {
  func::FuncOp fn;
  int numStages;
  int64_t reqBM, reqBN, reqBK;
  int64_t padK; // 16-bit padding on the smem K-tile leading dim.
  int64_t warpsN;
  int64_t groupM;
  int64_t splitKParts;

  int64_t BM = 64, BN = 128;
  int64_t BK = 32; // K staging step (= 2 k16 mma sub-tiles); autotuner knob.
  int64_t BKs = 32; // padded smem leading dim = BK + padK (>= BK).
  int64_t M = 0, N = 0, K = 0;
  unsigned aArgIdx = 0, wArgIdx = 0;
  ContractionOp con;
  Type inputStorageType, f32;
  SmallVector<Value> retVals;
  bool hasEpilogue = false;
  bool hasRegisterBiasSavedEpilogue = false;
  unsigned biasArgIdx = 0;
  SmallVector<Role> argRoles;
  DenseMap<Value, Value> argMemref;

  ContractionMMALowering(func::FuncOp fn, int numStages, int64_t reqBM,
                         int64_t reqBN, int64_t reqBK, int64_t padK,
                         int64_t warpsN, int64_t groupM,
                         int64_t splitKParts)
      : fn(fn), numStages(numStages), reqBM(reqBM), reqBN(reqBN),
        reqBK(reqBK), padK(padK), warpsN(warpsN),
        groupM(groupM), splitKParts(splitKParts) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() == 0)
      return failure();
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    ContractionOp c;
    for (Operation &op : body.without_terminator()) {
      if (auto cc = dyn_cast<ContractionOp>(&op)) {
        if (c)
          return failure();
        c = cc;
      } else if (!isa<CastOp, MapUnaryOp, MapBinaryOp, SelectOp,
                           arith::ConstantOp>(&op)) {
        return failure();
      }
    }
    if (!c)
      return failure();
    con = c;
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() == 0 ||
        ret.getNumOperands() != fty.getNumResults())
      return failure();
    retVals.assign(ret.getOperands().begin(), ret.getOperands().end());
    hasEpilogue = retVals.size() != 1 || retVals.front() != c.getResult();
    if (!isRowByRowContraction(c.getSubscripts()))
      return failure();

    // Canonical live-linear graphs make an exact f16/bf16->f32 widening
    // explicit before the contraction. Peel only that lossless cast so the
    // tensor-core kernel still reads the original 16-bit ABI operands and
    // accumulates in f32. Direct low-precision operands remain supported.
    auto peelLowPrecisionInput = [&](Value operand) -> BlockArgument {
      if (auto arg = dyn_cast<BlockArgument>(operand))
        return arg;
      auto castOp = operand.getDefiningOp<CastOp>();
      if (!castOp)
        return {};
      auto srcTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
      auto dstTy = dyn_cast<RankedTensorType>(castOp.getResult().getType());
      if (!srcTy || !dstTy || srcTy.getShape() != dstTy.getShape() ||
          !isa<Float16Type, BFloat16Type>(srcTy.getElementType()) ||
          !isa<Float32Type>(dstTy.getElementType()))
        return {};
      return dyn_cast<BlockArgument>(castOp.getInput());
    };
    auto aArg = peelLowPrecisionInput(c.getA());
    auto wArg = peelLowPrecisionInput(c.getB());
    if (!aArg || !wArg || aArg.getOwner() != &body || wArg.getOwner() != &body)
      return failure();
    aArgIdx = aArg.getArgNumber();
    wArgIdx = wArg.getArgNumber();

    auto aTy = dyn_cast<RankedTensorType>(aArg.getType());
    auto wTy = dyn_cast<RankedTensorType>(wArg.getType());
    auto conATy = dyn_cast<RankedTensorType>(c.getA().getType());
    auto conWTy = dyn_cast<RankedTensorType>(c.getB().getType());
    auto yTy = dyn_cast<RankedTensorType>(c.getResult().getType());
    if (!aTy || !wTy || !conATy || !conWTy || !yTy ||
        !aTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !conATy.hasStaticShape() || !conWTy.hasStaticShape() ||
        !yTy.hasStaticShape())
      return failure();
    if (aTy.getRank() != 2 || wTy.getRank() != 2 || yTy.getRank() != 2)
      return failure();
    if (!isa<Float16Type, BFloat16Type>(aTy.getElementType()) ||
        wTy.getElementType() != aTy.getElementType() ||
        !isa<Float32Type>(yTy.getElementType()))
      return failure();
    inputStorageType = aTy.getElementType();
    if (conATy.getShape() != aTy.getShape() ||
        conWTy.getShape() != wTy.getShape() ||
        (conATy.getElementType() != inputStorageType &&
         !isa<Float32Type>(conATy.getElementType())) ||
        (conWTy.getElementType() != inputStorageType &&
         !isa<Float32Type>(conWTy.getElementType())))
      return failure();
    f32 = yTy.getElementType();

    M = aTy.getDimSize(0);
    K = aTy.getDimSize(1);
    N = wTy.getDimSize(0);
    if (wTy.getDimSize(1) != K || yTy.getDimSize(0) != M ||
        yTy.getDimSize(1) != N)
      return failure();
    for (auto [index, value] : llvm::enumerate(retVals)) {
      auto resultTy = dyn_cast<RankedTensorType>(value.getType());
      auto declaredTy = dyn_cast<RankedTensorType>(fty.getResult(index));
      if (!resultTy || !declaredTy || !resultTy.hasStaticShape() ||
          resultTy != declaredTy || resultTy.getRank() != 2 ||
          resultTy.getDimSize(0) != M || resultTy.getDimSize(1) != N ||
          (resultTy.getElementType() != inputStorageType &&
           resultTy.getElementType() != f32))
        return failure();
    }

    // Exact captured fast epilogue: both external low-precision results are
    // casts of the same f32 (contraction + widened same-storage Col bias) value.
    // The register-level
    // mma.sync accumulator has a known lane layout, so this case can add bias,
    // round once, and store to both destinations directly from registers.  It
    // avoids the generic activation DAG's full f32 shared spill + barrier while
    // preserving the unchanged live graph and its saved-activation ABI.
    if (retVals.size() == 2) {
      auto firstCast = retVals[0].getDefiningOp<CastOp>();
      auto secondCast = retVals[1].getDefiningOp<CastOp>();
      if (firstCast && secondCast &&
          firstCast.getInput() == secondCast.getInput() &&
          cast<RankedTensorType>(firstCast.getResult().getType())
                  .getElementType() == inputStorageType &&
          cast<RankedTensorType>(secondCast.getResult().getType())
                  .getElementType() == inputStorageType) {
        auto add = firstCast.getInput().getDefiningOp<MapBinaryOp>();
        if (add && add.getFn() == "add") {
          Value other;
          if (add.getLhs() == c.getResult())
            other = add.getRhs();
          else if (add.getRhs() == c.getResult())
            other = add.getLhs();
          auto biasCast = other ? other.getDefiningOp<CastOp>() : CastOp();
          auto biasArg =
              biasCast ? dyn_cast<BlockArgument>(biasCast.getInput())
                       : BlockArgument();
          auto biasTy = biasArg
                            ? dyn_cast<RankedTensorType>(biasArg.getType())
                            : RankedTensorType();
          auto widenedTy = biasCast
                               ? dyn_cast<RankedTensorType>(
                                     biasCast.getResult().getType())
                               : RankedTensorType();
          if (biasArg && biasArg.getOwner() == &body && biasTy && widenedTy &&
              biasTy.hasStaticShape() && biasTy.getRank() == 1 &&
              biasTy.getDimSize(0) == N && (N % 2) == 0 &&
              biasTy.getElementType() == inputStorageType &&
              widenedTy.getShape() == biasTy.getShape() &&
              isa<Float32Type>(widenedTy.getElementType())) {
            hasRegisterBiasSavedEpilogue = true;
            biasArgIdx = biasArg.getArgNumber();
          }
        }
      }
    }

    // BK must be a positive multiple of the MMA K step (m16n8k16); reject an
    // illegal knob value rather than silently mis-tiling.
    BK = reqBK;
    if (BK <= 0 || BK % MMA_K != 0)
      return failure();
    BM = reqBM;
    BN = reqBN;
    if (BM <= 0 || BN <= 0)
      return failure();
    if (M % BM != 0 || N % BN != 0) {
      BM = 64;
      BN = 64;
    } else if (BN > 64 && N % 64 == 0) {
      int64_t wideBlocks = (M / BM) * (N / BN);
      if (wideBlocks < 64)
        BN = 64;
    }
    if (K % BK != 0 || M % BM != 0 || N % BN != 0)
      return failure();
    if (BM % MMA_M != 0 || BN % MMA_N != 0)
      return failure();
    // Each thread issues complete 128-bit cp.async packets. A's packet count
    // is integral because BK is a multiple of MMA_K; require the same for W
    // rather than silently leaving a suffix of the staged tile uncopied.
    if (warpsN != 1 && warpsN != 2 && warpsN != 4 && warpsN != 8)
      return failure();
    if (groupM < 1 || groupM > 32 || (groupM & (groupM - 1)) != 0)
      return failure();
    if (BN % (MMA_N * warpsN) != 0 ||
        (BM / MMA_M) * warpsN * 32 > 1024)
      return failure();
    int64_t copyThreads = (BM / MMA_M) * warpsN * 32;
    int64_t aPackets = (BM * BK) / 8;
    int64_t wPackets = (BN * BK) / 8;
    bool hasPacketTail = aPackets % copyThreads != 0 ||
                         wPackets % copyThreads != 0;
    if (hasPacketTail && !(padK >= 8 && padK % 8 == 0))
      return failure();
    if (splitKParts != 1 && splitKParts != 2 && splitKParts != 4)
      return failure();
    if (K % (BK * splitKParts) != 0)
      return failure();
    if (splitKParts > 1 && !hasRegisterBiasSavedEpilogue)
      return failure();
    // Padded smem leading dim: shift each K-tile row by padK 16-bit elements so
    // ldmatrix reads land on distinct shared-memory banks. padK must be a
    // multiple of 8 (one 128-bit cp.async packet); an illegal knob disables
    // padding rather than misaligning the copy.
    BKs = BK;
    if (padK > 0 && padK % 8 == 0)
      BKs = BK + padK;

    auto inputs = fty.getInputs();
    argRoles.assign(inputs.size(), Role::Row);
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (i == aArgIdx || i == wArgIdx) {
        argRoles[i] = Role::Row;
        continue;
      }
      auto t = dyn_cast<RankedTensorType>(inputs[i]);
      if (!t || !t.hasStaticShape() ||
          (t.getElementType() != inputStorageType &&
           !isa<Float32Type>(t.getElementType())))
        return failure();
      if (t.getRank() == 1 && t.getDimSize(0) == N)
        argRoles[i] = Role::Col;
      else if (t.getRank() == 0)
        argRoles[i] = Role::Scalar;
      else
        return failure();
    }
    return success();
  }

  // ---- epilogue scalar evaluator (identical to the WMMA pass) ------------ //
  Value emitEpilogue(OpBuilder &b, Location loc, Value v, Value cScalar,
                     Value gcol) {
    if (v == con.getResult())
      return cScalar;
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Col)
        return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{gcol});
      if (role == Role::Scalar)
        return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{});
      return nullptr;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
        if (dense.isSplat())
          return b.create<arith::ConstantOp>(
              loc, b.getF32FloatAttr(
                       dense.getSplatValue<APFloat>().convertToDouble()));
      return nullptr;
    }
    if (auto castOp = dyn_cast<CastOp>(def)) {
      Value source =
          emitEpilogue(b, loc, castOp.getInput(), cScalar, gcol);
      if (!source)
        return nullptr;
      auto resultTy = cast<RankedTensorType>(castOp.getResult().getType());
      Type target = resultTy.getElementType();
      if (source.getType() == target)
        return source;
      auto sourceFloat = dyn_cast<FloatType>(source.getType());
      auto targetFloat = dyn_cast<FloatType>(target);
      if (!sourceFloat || !targetFloat)
        return nullptr;
      if (sourceFloat.getWidth() < targetFloat.getWidth())
        return b.create<arith::ExtFOp>(loc, target, source);
      if (sourceFloat.getWidth() > targetFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, target, source);
      return nullptr;
    }
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return emitScalarUnary(b, loc, mu.getFn(),
                             emitEpilogue(b, loc, mu.getInput(), cScalar, gcol));
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      Value lhs = emitEpilogue(b, loc, mb.getLhs(), cScalar, gcol);
      Value rhs = emitEpilogue(b, loc, mb.getRhs(), cScalar, gcol);
      if (!lhs || !rhs)
        return nullptr;
      if (isCmpFn(mb.getFn()))
        return emitScalarCmp(b, loc, mb.getFn(), lhs, rhs);
      return emitScalarBinary(b, loc, mb.getFn(), lhs, rhs);
    }
    if (auto sel = dyn_cast<SelectOp>(def)) {
      Value cnd = emitEpilogue(b, loc, sel.getCond(), cScalar, gcol);
      Value x = emitEpilogue(b, loc, sel.getX(), cScalar, gcol);
      Value y = emitEpilogue(b, loc, sel.getY(), cScalar, gcol);
      if (!cnd || !x || !y)
        return nullptr;
      return b.create<arith::SelectOp>(loc, cnd, x, y);
    }
    return nullptr;
  }

  /// Split the K reduction into independent mma.sync CTAs, materialize exact
  /// f32 partials in device scratch, then launch an ordered additive finalize
  /// that applies the certified bias/cast epilogue and writes both observables.
  LogicalResult rewriteSplitK() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    SmallVector<Type> argumentTypes;
    for (Type type : fn.getFunctionType().getInputs()) {
      auto tensor = cast<RankedTensorType>(type);
      argumentTypes.push_back(
          MemRefType::get(tensor.getShape(), tensor.getElementType()));
    }
    for (Type type : fn.getFunctionType().getResults()) {
      auto tensor = cast<RankedTensorType>(type);
      argumentTypes.push_back(
          MemRefType::get(tensor.getShape(), tensor.getElementType()));
    }
    auto loweredType = FunctionType::get(ctx, argumentTypes, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto lowered = b.create<func::FuncOp>(loc, name, loweredType);
    Block *entry = lowered.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value aMem = entry->getArgument(aArgIdx);
    Value wMem = entry->getArgument(wArgIdx);
    Value biasMem = entry->getArgument(biasArgIdx);
    unsigned outputBase = fn.getFunctionType().getNumInputs();
    SmallVector<Value> outputMems;
    for (unsigned index = 0; index < fn.getFunctionType().getNumResults();
         ++index)
      outputMems.push_back(entry->getArgument(outputBase + index));

    auto partialType = MemRefType::get({splitKParts, M, N}, f32);
    Value partials = b.create<memref::AllocOp>(loc, partialType);

    int64_t numWarpsM = BM / MMA_M;
    int64_t numWarps = numWarpsM * warpsN;
    int64_t blockThreads = numWarps * 32;
    int64_t warpTileN = BN / warpsN;
    int64_t nFrag = warpTileN / MMA_N;
    int64_t gridN = N / BN;
    int64_t gridM = M / BM;
    int64_t tileBlocks = gridM * gridN;
    int64_t partialGrid = tileBlocks * splitKParts;
    int64_t localK = K / splitKParts;
    int64_t numK = localK / BK;
    int nStages = std::min<int64_t>(static_cast<int64_t>(numStages), numK);
    if (nStages < 1)
      nStages = 1;

    int64_t aBytes = BM * BKs * 2;
    int64_t wBytes = BN * BKs * 2;
    int64_t aAllBytes = nStages * aBytes;
    int64_t wAllBytes = nStages * wBytes;
    int64_t sharedBytes = aAllBytes + wAllBytes;

    Schedule schedule;
    schedule.gridX = partialGrid;
    schedule.blockX = blockThreads;
    schedule.smemBytes = sharedBytes;
    schedule.pipelineDepth = nStages;
    schedule.perThreadTile = nFrag;
    schedule.splitFactor = splitKParts;
    ::intent::attachLaunchPlan(lowered, schedule);

    Value cPartialGrid =
        b.create<arith::ConstantIndexOp>(loc, partialGrid);
    Value cBlockThreads =
        b.create<arith::ConstantIndexOp>(loc, blockThreads);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value cSharedBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr(static_cast<int32_t>(sharedBytes)));
    auto partialLaunch = b.create<gpu::LaunchOp>(
        loc, cPartialGrid, c1, c1, cBlockThreads, c1, c1, cSharedBytes);
    b.setInsertionPointToStart(&partialLaunch.getBody().front());

    Value bx = partialLaunch.getBlockIds().x;
    Value tx = partialLaunch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cWarpsN = b.create<arith::ConstantIndexOp>(loc, warpsN);
    Value cWarpTileN = b.create<arith::ConstantIndexOp>(loc, warpTileN);
    Value cBK = b.create<arith::ConstantIndexOp>(loc, BK);
    Value cParts = b.create<arith::ConstantIndexOp>(loc, splitKParts);
    Value split = b.create<arith::RemUIOp>(loc, bx, cParts);
    Value tileBlock = b.create<arith::DivUIOp>(loc, bx, cParts);
    Value cLocalK = b.create<arith::ConstantIndexOp>(loc, localK);
    Value kOrigin = b.create<arith::MulIOp>(loc, split, cLocalK);

    Value cGroupM = b.create<arith::ConstantIndexOp>(loc, groupM);
    Value cGridM = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cWidth = b.create<arith::ConstantIndexOp>(loc, groupM * gridN);
    Value groupId = b.create<arith::DivUIOp>(loc, tileBlock, cWidth);
    Value pidInW = b.create<arith::RemUIOp>(loc, tileBlock, cWidth);
    Value gmStart = b.create<arith::MulIOp>(loc, groupId, cGroupM);
    Value remM = b.create<arith::SubIOp>(loc, cGridM, gmStart);
    Value groupSize = b.create<arith::MinUIOp>(loc, remM, cGroupM);
    Value pidM = b.create<arith::AddIOp>(
        loc, gmStart, b.create<arith::RemUIOp>(loc, tileBlock, groupSize));
    Value pidN = b.create<arith::DivUIOp>(loc, pidInW, groupSize);
    Value cBM = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBN = b.create<arith::ConstantIndexOp>(loc, BN);
    Value rowBlock = b.create<arith::MulIOp>(loc, pidM, cBM);
    Value colBlock = b.create<arith::MulIOp>(loc, pidN, cBN);

    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value warpM = b.create<arith::DivUIOp>(loc, warp, cWarpsN);
    Value warpN = b.create<arith::RemUIOp>(loc, warp, cWarpsN);
    Value lane = b.create<arith::RemUIOp>(loc, tx, c32);
    Value warpRowBase = b.create<arith::MulIOp>(loc, warpM, c16);
    Value warpColBase = b.create<arith::MulIOp>(loc, warpN, cWarpTileN);
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);
    Value laneDiv16 = b.create<arith::DivUIOp>(loc, lane, c16);
    Value aColLane = b.create<arith::MulIOp>(loc, laneDiv16, c8);
    Value aRow = b.create<arith::AddIOp>(loc, warpRowBase, laneMod16);
    Value laneMod8 = b.create<arith::RemUIOp>(loc, lane, c8);
    Value laneDiv8 = b.create<arith::DivUIOp>(loc, lane, c8);
    Value wColLane = b.create<arith::MulIOp>(loc, laneDiv8, c8);
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value storeCol = b.create<arith::MulIOp>(loc, laneMod4, c2);

    auto rawType = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                   nullptr, wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawType);
    auto aSharedType =
        MemRefType::get({nStages, BM, BKs}, inputStorageType, nullptr, wgSpace);
    auto wSharedType =
        MemRefType::get({nStages, BN, BKs}, inputStorageType, nullptr, wgSpace);
    Value aShared =
        b.create<memref::ViewOp>(loc, aSharedType, raw, c0, ValueRange{});
    Value wOffset = b.create<arith::ConstantIndexOp>(loc, aAllBytes);
    Value wShared = b.create<memref::ViewOp>(loc, wSharedType, raw, wOffset,
                                             ValueRange{});

    auto aFragmentType = VectorType::get({4, 2}, inputStorageType);
    auto wFragmentType = VectorType::get({2, 2}, inputStorageType);
    auto accumulatorType = VectorType::get({2, 2}, f32);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroAccumulator = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accumulatorType,
                                    b.getF32FloatAttr(0.0).getValue()));
    SmallVector<Value> initialAccumulators(nFrag, zeroAccumulator);

    auto tokenType = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr destinationElements = b.getIndexAttr(8);
    UnitAttr bypassL1 = b.getUnitAttr();
    int64_t columnsPerRow = BK / 8;
    int64_t vectorsPerThreadA = (BM * BK) / 8 / blockThreads;
    int64_t vectorsPerThreadW = (BN * BK) / 8 / blockThreads;
    Value cColumnsPerRow =
        b.create<arith::ConstantIndexOp>(loc, columnsPerRow);
    Value cNStages = b.create<arith::ConstantIndexOp>(loc, nStages);
    Value cNStagesMinusOne =
        b.create<arith::ConstantIndexOp>(loc, nStages - 1);

    auto loadTile = [&](OpBuilder &nested, Location nestedLoc, Value localKTile,
                        Value stage) -> Value {
      SmallVector<Value> tokens;
      for (int64_t iteration = 0; iteration < vectorsPerThreadA; ++iteration) {
        Value base = nested.create<arith::ConstantIndexOp>(
            nestedLoc, iteration * blockThreads);
        Value vectorIndex =
            nested.create<arith::AddIOp>(nestedLoc, tx, base);
        Value row = nested.create<arith::DivUIOp>(nestedLoc, vectorIndex,
                                                   cColumnsPerRow);
        Value columnGroup = nested.create<arith::RemUIOp>(
            nestedLoc, vectorIndex, cColumnsPerRow);
        Value column =
            nested.create<arith::MulIOp>(nestedLoc, columnGroup, c8);
        Value globalK = nested.create<arith::AddIOp>(
            nestedLoc, kOrigin,
            nested.create<arith::AddIOp>(nestedLoc, localKTile, column));
        Value globalRow =
            nested.create<arith::AddIOp>(nestedLoc, rowBlock, row);
        tokens.push_back(nested.create<nvgpu::DeviceAsyncCopyOp>(
            nestedLoc, tokenType, aShared, ValueRange{stage, row, column}, aMem,
            ValueRange{globalRow, globalK}, destinationElements,
            /*srcElements=*/Value(), bypassL1));
      }
      for (int64_t iteration = 0; iteration < vectorsPerThreadW; ++iteration) {
        Value base = nested.create<arith::ConstantIndexOp>(
            nestedLoc, iteration * blockThreads);
        Value vectorIndex =
            nested.create<arith::AddIOp>(nestedLoc, tx, base);
        Value row = nested.create<arith::DivUIOp>(nestedLoc, vectorIndex,
                                                   cColumnsPerRow);
        Value columnGroup = nested.create<arith::RemUIOp>(
            nestedLoc, vectorIndex, cColumnsPerRow);
        Value column =
            nested.create<arith::MulIOp>(nestedLoc, columnGroup, c8);
        Value globalK = nested.create<arith::AddIOp>(
            nestedLoc, kOrigin,
            nested.create<arith::AddIOp>(nestedLoc, localKTile, column));
        Value globalRow =
            nested.create<arith::AddIOp>(nestedLoc, colBlock, row);
        tokens.push_back(nested.create<nvgpu::DeviceAsyncCopyOp>(
            nestedLoc, tokenType, wShared, ValueRange{stage, row, column}, wMem,
            ValueRange{globalRow, globalK}, destinationElements,
            /*srcElements=*/Value(), bypassL1));
      }
      return nested.create<nvgpu::DeviceAsyncCreateGroupOp>(nestedLoc,
                                                             tokenType, tokens);
    };

    auto computeTile = [&](OpBuilder &nested, Location nestedLoc, Value stage,
                           ValueRange accumulators) -> SmallVector<Value> {
      SmallVector<Value> current(accumulators.begin(), accumulators.end());
      for (int64_t kStep = 0; kStep < BK / MMA_K; ++kStep) {
        Value kConstant =
            nested.create<arith::ConstantIndexOp>(nestedLoc, kStep * MMA_K);
        Value aKColumn =
            nested.create<arith::AddIOp>(nestedLoc, kConstant, aColLane);
        Value aFragment = nested.create<nvgpu::LdMatrixOp>(
            nestedLoc, aFragmentType, aShared,
            ValueRange{stage, aRow, aKColumn}, /*transpose=*/false,
            /*numTiles=*/4);
        Value wKColumn =
            nested.create<arith::AddIOp>(nestedLoc, kConstant, wColLane);
        for (int fragment = 0; fragment < nFrag; ++fragment) {
          Value nBase = nested.create<arith::AddIOp>(
              nestedLoc, warpColBase,
              nested.create<arith::ConstantIndexOp>(nestedLoc,
                                                     fragment * MMA_N));
          Value wRow =
              nested.create<arith::AddIOp>(nestedLoc, nBase, laneMod8);
          Value wFragment = nested.create<nvgpu::LdMatrixOp>(
              nestedLoc, wFragmentType, wShared,
              ValueRange{stage, wRow, wKColumn}, /*transpose=*/false,
              /*numTiles=*/2);
          current[fragment] = nested.create<nvgpu::MmaSyncOp>(
              nestedLoc, aFragment, wFragment, current[fragment], mmaShape);
        }
      }
      return current;
    };

    Value prologueToken;
    for (int stageIndex = 0; stageIndex < nStages - 1; ++stageIndex) {
      Value localKTile =
          b.create<arith::ConstantIndexOp>(loc, stageIndex * BK);
      Value stage = b.create<arith::ConstantIndexOp>(loc, stageIndex);
      prologueToken = loadTile(b, loc, localKTile, stage);
    }
    int64_t steadyTrips = numK - nStages + 1;
    Value cSteadyTrips = b.create<arith::ConstantIndexOp>(loc, steadyTrips);
    IntegerAttr waitCount = b.getI32IntegerAttr(nStages - 1);
    auto mainLoop = b.create<scf::ForOp>(
        loc, c0, cSteadyTrips, c1, initialAccumulators,
        [&](OpBuilder &nested, Location nestedLoc, Value iteration,
            ValueRange accumulators) {
          Value nextTile = nested.create<arith::AddIOp>(
              nestedLoc, iteration, cNStagesMinusOne);
          Value nextK =
              nested.create<arith::MulIOp>(nestedLoc, nextTile, cBK);
          Value writeStage =
              nested.create<arith::RemUIOp>(nestedLoc, nextTile, cNStages);
          Value group = loadTile(nested, nestedLoc, nextK, writeStage);
          nested.create<nvgpu::DeviceAsyncWaitOp>(nestedLoc, group, waitCount);
          nested.create<gpu::BarrierOp>(nestedLoc);
          Value readStage =
              nested.create<arith::RemUIOp>(nestedLoc, iteration, cNStages);
          SmallVector<Value> current =
              computeTile(nested, nestedLoc, readStage, accumulators);
          nested.create<gpu::BarrierOp>(nestedLoc);
          nested.create<scf::YieldOp>(nestedLoc, current);
        });
    SmallVector<Value> finalAccumulators(mainLoop.getResults().begin(),
                                         mainLoop.getResults().end());
    for (int tail = 0; tail < nStages - 1; ++tail) {
      int64_t tile = numK - nStages + 1 + tail;
      Value readStage =
          b.create<arith::ConstantIndexOp>(loc, tile % nStages);
      b.create<nvgpu::DeviceAsyncWaitOp>(
          loc, prologueToken, b.getI32IntegerAttr(nStages - 2 - tail));
      b.create<gpu::BarrierOp>(loc);
      finalAccumulators =
          computeTile(b, loc, readStage, finalAccumulators);
    }

    Value globalWarpRow =
        b.create<arith::AddIOp>(loc, rowBlock, warpRowBase);
    for (int fragment = 0; fragment < nFrag; ++fragment) {
      Value nBase = b.create<arith::AddIOp>(
          loc, warpColBase,
          b.create<arith::ConstantIndexOp>(loc, fragment * MMA_N));
      Value columnBase = b.create<arith::AddIOp>(loc, colBlock, nBase);
      for (int rowIndex = 0; rowIndex < 2; ++rowIndex) {
        Value rowOffset =
            b.create<arith::ConstantIndexOp>(loc, rowIndex * 8);
        Value row = b.create<arith::AddIOp>(
            loc, globalWarpRow,
            b.create<arith::AddIOp>(loc, laneDiv4, rowOffset));
        for (int columnIndex = 0; columnIndex < 2; ++columnIndex) {
          Value columnOffset =
              b.create<arith::ConstantIndexOp>(loc, columnIndex);
          Value column = b.create<arith::AddIOp>(
              loc, columnBase,
              b.create<arith::AddIOp>(loc, storeCol, columnOffset));
          Value value = b.create<vector::ExtractOp>(
              loc, finalAccumulators[fragment],
              ArrayRef<int64_t>{static_cast<int64_t>(rowIndex),
                                static_cast<int64_t>(columnIndex)});
          b.create<memref::StoreOp>(loc, value, partials,
                                    ValueRange{split, row, column});
        }
      }
    }
    b.create<gpu::TerminatorOp>(loc);

    b.setInsertionPointAfter(partialLaunch);
    constexpr int64_t finalizeThreads = 256;
    int64_t finalizeElements = M * N;
    int64_t finalizeGrid =
        (finalizeElements + finalizeThreads - 1) / finalizeThreads;
    Value cFinalizeGrid =
        b.create<arith::ConstantIndexOp>(loc, finalizeGrid);
    Value cFinalizeThreads =
        b.create<arith::ConstantIndexOp>(loc, finalizeThreads);
    auto finalizeLaunch = b.create<gpu::LaunchOp>(
        loc, cFinalizeGrid, c1, c1, cFinalizeThreads, c1, c1);
    b.setInsertionPointToStart(&finalizeLaunch.getBody().front());
    Value finalBlock = finalizeLaunch.getBlockIds().x;
    Value finalThread = finalizeLaunch.getThreadIds().x;
    Value cFinalizeStride =
        b.create<arith::ConstantIndexOp>(loc, finalizeThreads);
    Value flat = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, finalBlock, cFinalizeStride),
        finalThread);
    Value cFinalizeElements =
        b.create<arith::ConstantIndexOp>(loc, finalizeElements);
    Value active = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, flat, cFinalizeElements);
    auto guarded = b.create<scf::IfOp>(loc, active,
                                       /*withElseRegion=*/false);
    OpBuilder bodyBuilder = guarded.getThenBodyBuilder();
    Value cN = bodyBuilder.create<arith::ConstantIndexOp>(loc, N);
    Value row = bodyBuilder.create<arith::DivUIOp>(loc, flat, cN);
    Value column = bodyBuilder.create<arith::RemUIOp>(loc, flat, cN);
    Value partialZero =
        bodyBuilder.create<arith::ConstantIndexOp>(loc, 0);
    Value sum = bodyBuilder.create<memref::LoadOp>(
        loc, partials, ValueRange{partialZero, row, column});
    for (int64_t part = 1; part < splitKParts; ++part) {
      Value partIndex =
          bodyBuilder.create<arith::ConstantIndexOp>(loc, part);
      Value next = bodyBuilder.create<memref::LoadOp>(
          loc, partials, ValueRange{partIndex, row, column});
      sum = bodyBuilder.create<arith::AddFOp>(loc, sum, next);
    }
    Value biasStorage =
        bodyBuilder.create<memref::LoadOp>(loc, biasMem, ValueRange{column});
    Value bias =
        bodyBuilder.create<arith::ExtFOp>(loc, f32, biasStorage);
    Value preactivation =
        bodyBuilder.create<arith::AddFOp>(loc, sum, bias);
    Value stored = bodyBuilder.create<arith::TruncFOp>(
        loc, inputStorageType, preactivation);
    for (Value output : outputMems)
      bodyBuilder.create<memref::StoreOp>(loc, stored, output,
                                          ValueRange{row, column});
    b.setInsertionPointAfter(guarded);
    b.create<gpu::TerminatorOp>(loc);

    b.setInsertionPointAfter(finalizeLaunch);
    b.create<memref::DeallocOp>(loc, partials);
    b.create<func::ReturnOp>(loc);

    auto i64 = IntegerType::get(ctx, 64);
    lowered->setAttr("intent.backend.lowering_path_kind",
                     StringAttr::get(ctx, "contraction"));
    lowered->setAttr("intent.backend.reduction_topology",
                     StringAttr::get(ctx, "multi_stage_partial_finalize"));
    lowered->setAttr("intent.backend.approximation_mode",
                     StringAttr::get(ctx, "none"));
    SmallVector<Attribute> splitTrace{
        StringAttr::get(ctx, "cuda.contraction.mma-sync.v1")};
    if (warpsN > 1)
      splitTrace.push_back(StringAttr::get(
          ctx, "cuda.contraction.mma-sync.warp-n-partition.v1"));
    splitTrace.push_back(StringAttr::get(
        ctx, "cuda.contraction.split_k_partial_finalize.v1"));
    lowered->setAttr("intent.backend.transformation_trace",
                     ArrayAttr::get(ctx, splitTrace));
    lowered->setAttr("intent.backend.stage_count", IntegerAttr::get(i64, 2));
    lowered->setAttr("intent.backend.reduction_parts",
                     IntegerAttr::get(i64, splitKParts));
    lowered->setAttr("intent.backend.warps_n",
                     IntegerAttr::get(i64, warpsN));
    lowered->setAttr("intent.backend.group_m",
                     IntegerAttr::get(i64, groupM));
    int64_t barrierCount = 0;
    lowered.walk([&](gpu::BarrierOp) { ++barrierCount; });
    lowered->setAttr("intent.backend.fma_count", IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.shuffle_count", IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.barrier_count",
                     IntegerAttr::get(i64, barrierCount));
    lowered->setAttr("intent.backend.all_reduce_count",
                     IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.block_threads",
                     IntegerAttr::get(i64, blockThreads));
    lowered->setAttr("intent.backend.grid_blocks",
                     IntegerAttr::get(i64, partialGrid));
    lowered->setAttr("intent.backend.predicated_value_if_count",
                     IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.folded_constexpr_select_count",
                     IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.vector_width", IntegerAttr::get(i64, 1));
    fn.erase();
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    if (splitKParts > 1)
      return rewriteSplitK();
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    SmallVector<Type> inTys;
    Type outputPairType = VectorType::get({2}, inputStorageType);
    for (auto [index, at] : llvm::enumerate(fn.getFunctionType().getInputs())) {
      auto t = cast<RankedTensorType>(at);
      if (hasRegisterBiasSavedEpilogue && index == biasArgIdx)
        inTys.push_back(MemRefType::get({N / 2}, outputPairType));
      else
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    for (Type resultType : fn.getFunctionType().getResults()) {
      auto t = cast<RankedTensorType>(resultType);
      if (hasRegisterBiasSavedEpilogue)
        inTys.push_back(MemRefType::get({M, N / 2}, outputPairType));
      else
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value aMem = entry->getArgument(aArgIdx);
    Value wMem = entry->getArgument(wArgIdx);
    unsigned outputBase = fn.getFunctionType().getNumInputs();
    SmallVector<Value> outputMems;
    for (unsigned i = 0; i < fn.getFunctionType().getNumResults(); ++i)
      outputMems.push_back(entry->getArgument(outputBase + i));
    Value yMem = outputMems.front();

    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);

    int64_t numWarpsM = BM / MMA_M;
    int64_t numWarps = numWarpsM * warpsN;
    int64_t groupThreads = numWarps * 32;
    int64_t blockThreads = groupThreads;
    int64_t warpTileN = BN / warpsN;
    int64_t nFrag = warpTileN / MMA_N; // n8 column tiles per warp

    int64_t gridN = N / BN, gridM = M / BM;
    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM * gridN);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, blockThreads);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    int64_t numK = K / BK;
    int nStages = std::min<int64_t>((int64_t)numStages, numK);
    if (nStages < 1)
      nStages = 1;

    int64_t aBytes = BM * BKs * 2, wBytes = BN * BKs * 2;
    int64_t aAllBytes = nStages * aBytes;
    int64_t wAllBytes = nStages * wBytes;
    int64_t cBytes = hasEpilogue && !hasRegisterBiasSavedEpilogue
                         ? BM * BN * 4
                         : 0;
    Value shBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr((int32_t)(aAllBytes + wAllBytes + cBytes)));

    // M6: emit the structured launch plan (grid/block/smem + the tuned tile
    // knobs) for the runtime, so it reads the tensor-core launch config from a
    // Schedule instead of regex-scraping the lowered mma.sync host module.
    Schedule sched;
    sched.gridX = gridM * gridN;
    sched.blockX = blockThreads;
    sched.smemBytes = aAllBytes + wAllBytes + cBytes;
    sched.pipelineDepth = nStages;
    sched.perThreadTile = nFrag;
    sched.vectorWidth = hasRegisterBiasSavedEpilogue ? 2 : 1;
    ::intent::attachLaunchPlan(gpuFn, sched);

    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value localTx = tx;
    Value partitionBase = c0;
    Value cBK = b.create<arith::ConstantIndexOp>(loc, BK);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value cWarpsN = b.create<arith::ConstantIndexOp>(loc, warpsN);
    Value cWarpTileN = b.create<arith::ConstantIndexOp>(loc, warpTileN);

    // Flat pid -> (pid_m, pid_n) with the GROUP_M L2 swizzle.
    Value cGroupM = b.create<arith::ConstantIndexOp>(loc, groupM);
    Value cGridM = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cWidth = b.create<arith::ConstantIndexOp>(loc, groupM * gridN);
    Value groupId = b.create<arith::DivUIOp>(loc, bx, cWidth);
    Value pidInW = b.create<arith::RemUIOp>(loc, bx, cWidth);
    Value gmStart = b.create<arith::MulIOp>(loc, groupId, cGroupM);
    Value remM = b.create<arith::SubIOp>(loc, cGridM, gmStart);
    Value groupSz = b.create<arith::MinUIOp>(loc, remM, cGroupM);
    Value pidM = b.create<arith::AddIOp>(
        loc, gmStart, b.create<arith::RemUIOp>(loc, bx, groupSz));
    Value pidN = b.create<arith::DivUIOp>(loc, pidInW, groupSz);
    Value cBMv = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBNv = b.create<arith::ConstantIndexOp>(loc, BN);
    Value rowBlk = b.create<arith::MulIOp>(loc, pidM, cBMv);
    Value colBlk = b.create<arith::MulIOp>(loc, pidN, cBNv);

    // Per-thread lane/warp decomposition + the ldmatrix/store lane offsets.
    Value warp = b.create<arith::DivUIOp>(loc, localTx, c32);
    Value warpM = b.create<arith::DivUIOp>(loc, warp, cWarpsN);
    Value warpN = b.create<arith::RemUIOp>(loc, warp, cWarpsN);
    Value lane = b.create<arith::RemUIOp>(loc, localTx, c32);
    Value warpRowBase =
        b.create<arith::MulIOp>(loc, warpM, c16); // 16-row strip
    Value warpColBase = b.create<arith::MulIOp>(loc, warpN, cWarpTileN);
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);
    Value laneDiv16 = b.create<arith::DivUIOp>(loc, lane, c16);
    Value aColLane = b.create<arith::MulIOp>(loc, laneDiv16, c8); // (lane/16)*8
    Value aRow = b.create<arith::AddIOp>(loc, warpRowBase, laneMod16);
    Value laneMod8 = b.create<arith::RemUIOp>(loc, lane, c8);
    Value laneDiv8 = b.create<arith::DivUIOp>(loc, lane, c8);
    Value wColLane = b.create<arith::MulIOp>(loc, laneDiv8, c8); // (lane/8)*8
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);    // store row grp
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value stCol = b.create<arith::MulIOp>(loc, laneMod4, c2); // (lane%4)*2

    // Dynamic-shared staging (rank-3 rotating buffers, as the WMMA pass).
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto aShTy = MemRefType::get({nStages, BM, BKs}, inputStorageType, nullptr,
                                 wgSpace);
    auto wShTy = MemRefType::get({nStages, BN, BKs}, inputStorageType, nullptr,
                                 wgSpace);
    Value aSh = b.create<memref::ViewOp>(loc, aShTy, raw, c0, ValueRange{});
    Value wOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes);
    Value wSh = b.create<memref::ViewOp>(loc, wShTy, raw, wOff, ValueRange{});
    Value cSh;
    if (hasEpilogue && !hasRegisterBiasSavedEpilogue) {
      auto cShTy = MemRefType::get({BM, BN}, f32, nullptr, wgSpace);
      Value cOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes + wAllBytes);
      cSh = b.create<memref::ViewOp>(loc, cShTy, raw, cOff, ValueRange{});
    }

    auto aFragTy = VectorType::get({4, 2}, inputStorageType);
    auto wFragTy = VectorType::get({2, 2}, inputStorageType);
    auto accTy = VectorType::get({2, 2}, f32);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});

    // nFrag accumulator vectors, init 0.
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));
    SmallVector<Value> accInit(nFrag, zeroAcc);

    // -- cp.async multistage staging (identical schedule to the WMMA pass) -- //
    auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr dstElems = b.getIndexAttr(8);
    UnitAttr bypassL1 = b.getUnitAttr();
    int64_t colsPerRow = BK / 8;
    int64_t aPackets = (BM * BK) / 8;
    int64_t wPackets = (BN * BK) / 8;
    int64_t vPerThreadA = (aPackets + groupThreads - 1) / groupThreads;
    int64_t vPerThreadW = (wPackets + groupThreads - 1) / groupThreads;
    Value cCPR = b.create<arith::ConstantIndexOp>(loc, colsPerRow);
    Value cAPackets = b.create<arith::ConstantIndexOp>(loc, aPackets);
    Value cWPackets = b.create<arith::ConstantIndexOp>(loc, wPackets);
    Value c8b = c8;
    Value cNStages = b.create<arith::ConstantIndexOp>(loc, nStages);
    Value cNm1 = b.create<arith::ConstantIndexOp>(loc, nStages - 1);

    auto loadTile = [&](OpBuilder &nb, Location nl, Value ktVal,
                        Value stageVal) -> Value {
      SmallVector<Value> toks;
      for (int64_t it = 0; it < vPerThreadA; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * groupThreads);
        Value vt = nb.create<arith::AddIOp>(nl, localTx, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8b);
        Value localK = nb.create<arith::AddIOp>(nl, ktVal, cc);
        Value gk = nb.create<arith::AddIOp>(nl, partitionBase, localK);
        Value ar = nb.create<arith::AddIOp>(nl, rowBlk, r);
        Value safeR = r, safeCc = cc, safeAr = ar, safeGk = gk;
        Value srcElements;
        if (aPackets % groupThreads != 0) {
          Value inRange = nb.create<arith::CmpIOp>(
              nl, arith::CmpIPredicate::ult, vt, cAPackets);
          safeR = nb.create<arith::SelectOp>(
              nl, inRange, r,
              nb.create<arith::RemUIOp>(nl, vt, cBMv));
          safeCc = nb.create<arith::SelectOp>(nl, inRange, cc, cBK);
          safeAr = nb.create<arith::SelectOp>(nl, inRange, ar, rowBlk);
          safeGk = nb.create<arith::SelectOp>(nl, inRange, gk, partitionBase);
          Value c8Elems = nb.create<arith::ConstantIndexOp>(nl, 8);
          Value c0Elems = nb.create<arith::ConstantIndexOp>(nl, 0);
          srcElements = nb.create<arith::SelectOp>(nl, inRange, c8Elems,
                                                   c0Elems);
        }
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, aSh, ValueRange{stageVal, safeR, safeCc}, aMem,
            ValueRange{safeAr, safeGk}, dstElems, srcElements, bypassL1));
      }
      for (int64_t it = 0; it < vPerThreadW; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * groupThreads);
        Value vt = nb.create<arith::AddIOp>(nl, localTx, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8b);
        Value localK = nb.create<arith::AddIOp>(nl, ktVal, cc);
        Value gk = nb.create<arith::AddIOp>(nl, partitionBase, localK);
        Value wr = nb.create<arith::AddIOp>(nl, colBlk, r);
        Value safeR = r, safeCc = cc, safeWr = wr, safeGk = gk;
        Value srcElements;
        if (wPackets % groupThreads != 0) {
          Value inRange = nb.create<arith::CmpIOp>(
              nl, arith::CmpIPredicate::ult, vt, cWPackets);
          safeR = nb.create<arith::SelectOp>(
              nl, inRange, r,
              nb.create<arith::RemUIOp>(nl, vt, cBNv));
          safeCc = nb.create<arith::SelectOp>(nl, inRange, cc, cBK);
          safeWr = nb.create<arith::SelectOp>(nl, inRange, wr, colBlk);
          safeGk = nb.create<arith::SelectOp>(nl, inRange, gk, partitionBase);
          Value c8Elems = nb.create<arith::ConstantIndexOp>(nl, 8);
          Value c0Elems = nb.create<arith::ConstantIndexOp>(nl, 0);
          srcElements = nb.create<arith::SelectOp>(nl, inRange, c8Elems,
                                                   c0Elems);
        }
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, wSh, ValueRange{stageVal, safeR, safeCc}, wMem,
            ValueRange{safeWr, safeGk}, dstElems, srcElements, bypassL1));
      }
      return nb.create<nvgpu::DeviceAsyncCreateGroupOp>(nl, tokTy, toks);
    };

    // Accumulate acc[nt] += A(stage) * W_nt(stage) over the BK/16 k-subtiles.
    auto computeTile = [&](OpBuilder &nb, Location nl, Value stageVal,
                           ValueRange acc) -> SmallVector<Value> {
      SmallVector<Value> cur(acc.begin(), acc.end());
      for (int64_t ks = 0; ks < BK / MMA_K; ++ks) {
        Value ksc = nb.create<arith::ConstantIndexOp>(nl, ks * MMA_K);
        Value aKcol = nb.create<arith::AddIOp>(nl, ksc, aColLane);
        Value aFrag = nb.create<nvgpu::LdMatrixOp>(
            nl, aFragTy, aSh, ValueRange{stageVal, aRow, aKcol},
            /*transpose=*/false, /*numTiles=*/4);
        Value wKcol = nb.create<arith::AddIOp>(nl, ksc, wColLane);
        for (int j = 0; j < nFrag; ++j) {
          Value nbase = nb.create<arith::AddIOp>(
              nl, warpColBase,
              nb.create<arith::ConstantIndexOp>(nl, j * MMA_N));
          Value wRow = nb.create<arith::AddIOp>(nl, nbase, laneMod8);
          Value wFrag = nb.create<nvgpu::LdMatrixOp>(
              nl, wFragTy, wSh, ValueRange{stageVal, wRow, wKcol},
              /*transpose=*/false, /*numTiles=*/2);
          cur[j] = nb.create<nvgpu::MmaSyncOp>(nl, aFrag, wFrag, cur[j],
                                               mmaShape);
        }
      }
      return cur;
    };

    // PROLOGUE.
    Value prologTok;
    for (int s = 0; s < nStages - 1; ++s) {
      Value ktVal = b.create<arith::ConstantIndexOp>(loc, s * BK);
      Value stage = b.create<arith::ConstantIndexOp>(loc, s);
      prologTok = loadTile(b, loc, ktVal, stage);
    }

    // STEADY STATE.
    int64_t steadyTrips = numK - nStages + 1;
    Value cSteady = b.create<arith::ConstantIndexOp>(loc, steadyTrips);
    IntegerAttr waitNm1 = b.getI32IntegerAttr(nStages - 1);
    auto forOp = b.create<scf::ForOp>(
        loc, c0, cSteady, c1, accInit,
        [&](OpBuilder &nb, Location nl, Value ki, ValueRange acc) {
          Value nextTile = nb.create<arith::AddIOp>(nl, ki, cNm1);
          Value ktNext = nb.create<arith::MulIOp>(nl, nextTile, cBK);
          Value stageW = nb.create<arith::RemUIOp>(nl, nextTile, cNStages);
          Value grp = loadTile(nb, nl, ktNext, stageW);
          nb.create<nvgpu::DeviceAsyncWaitOp>(nl, grp, waitNm1);
          nb.create<gpu::BarrierOp>(nl);
          Value stageR = nb.create<arith::RemUIOp>(nl, ki, cNStages);
          SmallVector<Value> cur = computeTile(nb, nl, stageR, acc);
          nb.create<gpu::BarrierOp>(nl);
          nb.create<scf::YieldOp>(nl, cur);
        });

    // EPILOGUE of the pipeline: the last nStages-1 (already prefetched) tiles.
    SmallVector<Value> finalAcc(forOp.getResults().begin(),
                                forOp.getResults().end());
    for (int j = 0; j < nStages - 1; ++j) {
      int64_t tile = numK - nStages + 1 + j;
      Value stageR = b.create<arith::ConstantIndexOp>(loc, tile % nStages);
      b.create<nvgpu::DeviceAsyncWaitOp>(loc, prologTok,
                                         b.getI32IntegerAttr(nStages - 2 - j));
      b.create<gpu::BarrierOp>(loc);
      finalAcc = computeTile(b, loc, stageR, finalAcc);
    }

    // Store. Textbook m16n8 accumulator layout:
    //   d[i][j] -> row = rowBlk + warp*16 + lane/4 + i*8,
    //              col = colBlk + nt*8 + (lane%4)*2 + j
    auto storeAcc = [&](OpBuilder &nb, Location nl, Value accVal, Value rowBase,
                        Value colBase, auto sink) {
      for (int i = 0; i < 2; ++i) {
        Value ic = nb.create<arith::ConstantIndexOp>(nl, i * 8);
        Value row =
            nb.create<arith::AddIOp>(nl, rowBase,
                                     nb.create<arith::AddIOp>(nl, laneDiv4, ic));
        for (int j = 0; j < 2; ++j) {
          Value jc = nb.create<arith::ConstantIndexOp>(nl, j);
          Value col = nb.create<arith::AddIOp>(
              nl, colBase, nb.create<arith::AddIOp>(nl, stCol, jc));
          Value v = nb.create<vector::ExtractOp>(
              nl, accVal, ArrayRef<int64_t>{(int64_t)i, (int64_t)j});
          sink(nb, nl, v, row, col);
        }
      }
    };

    Value warpRowGlobal = b.create<arith::AddIOp>(loc, rowBlk, warpRowBase);
    auto emitRegisterBiasSavedEpilogue =
        [&](OpBuilder &nb, Location nl, ValueRange accumulators) {
          Value biasMem = entry->getArgument(biasArgIdx);
          for (int j = 0; j < nFrag; ++j) {
            Value nbase = nb.create<arith::AddIOp>(
                nl, warpColBase,
                nb.create<arith::ConstantIndexOp>(nl, j * MMA_N));
            Value colBase = nb.create<arith::AddIOp>(nl, colBlk, nbase);
            Value firstCol = nb.create<arith::AddIOp>(nl, colBase, stCol);
            Value colPair = nb.create<arith::DivUIOp>(nl, firstCol, c2);
            Value biasPair =
                nb.create<memref::LoadOp>(nl, biasMem, ValueRange{colPair});
            for (int i = 0; i < 2; ++i) {
              Value ic = nb.create<arith::ConstantIndexOp>(nl, i * 8);
              Value row = nb.create<arith::AddIOp>(
                  nl, warpRowGlobal,
                  nb.create<arith::AddIOp>(nl, laneDiv4, ic));
              SmallVector<Value> storedLanes;
              for (int laneIndex = 0; laneIndex < 2; ++laneIndex) {
                Value value = nb.create<vector::ExtractOp>(
                    nl, accumulators[j],
                    ArrayRef<int64_t>{static_cast<int64_t>(i),
                                      static_cast<int64_t>(laneIndex)});
                Value biasStorage = nb.create<vector::ExtractOp>(
                    nl, biasPair, ArrayRef<int64_t>{laneIndex});
                Value biasF32 =
                    nb.create<arith::ExtFOp>(nl, f32, biasStorage);
                Value preactivation =
                    nb.create<arith::AddFOp>(nl, value, biasF32);
                storedLanes.push_back(nb.create<arith::TruncFOp>(
                    nl, inputStorageType, preactivation));
              }
              Value storedPair = nb.create<vector::FromElementsOp>(
                  nl, cast<VectorType>(outputPairType), storedLanes);
              for (Value output : outputMems)
                nb.create<memref::StoreOp>(nl, storedPair, output,
                                           ValueRange{row, colPair});
            }
          }
        };

    if (!hasEpilogue) {
      for (int j = 0; j < nFrag; ++j) {
        Value nbase = b.create<arith::AddIOp>(
            loc, warpColBase,
            b.create<arith::ConstantIndexOp>(loc, j * MMA_N));
        Value colBase = b.create<arith::AddIOp>(loc, colBlk, nbase);
        storeAcc(b, loc, finalAcc[j], warpRowGlobal, colBase,
                 [&](OpBuilder &nb, Location nl, Value v, Value row, Value col) {
                   nb.create<memref::StoreOp>(nl, v, yMem, ValueRange{row, col});
                 });
      }
    } else if (hasRegisterBiasSavedEpilogue) {
      emitRegisterBiasSavedEpilogue(b, loc, finalAcc);
    } else {
      // Spill the accumulators to the [BM,BN] shared tile, then cooperatively
      // reload + evaluate the bias/activation epilogue + write Y.
      Value warpRowLocal = warpRowBase; // block-local row
      for (int j = 0; j < nFrag; ++j) {
        Value nbase = b.create<arith::AddIOp>(
            loc, warpColBase,
            b.create<arith::ConstantIndexOp>(loc, j * MMA_N));
        storeAcc(b, loc, finalAcc[j], warpRowLocal, nbase,
                 [&](OpBuilder &nb, Location nl, Value v, Value row, Value col) {
                   nb.create<memref::StoreOp>(nl, v, cSh, ValueRange{row, col});
                 });
      }
      b.create<gpu::BarrierOp>(loc);
      Value cBN = b.create<arith::ConstantIndexOp>(loc, BN);
      int64_t epIters = (BM * BN) / groupThreads;
      for (int64_t it = 0; it < epIters; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * groupThreads);
        Value i = b.create<arith::AddIOp>(loc, localTx, base);
        Value r = b.create<arith::DivUIOp>(loc, i, cBN);
        Value cc = b.create<arith::RemUIOp>(loc, i, cBN);
        Value cScalar = b.create<memref::LoadOp>(loc, cSh, ValueRange{r, cc});
        Value gcol = b.create<arith::AddIOp>(loc, colBlk, cc);
        Value grow = b.create<arith::AddIOp>(loc, rowBlk, r);
        for (unsigned outputIndex = 0; outputIndex < retVals.size();
             ++outputIndex) {
          Value out =
              emitEpilogue(b, loc, retVals[outputIndex], cScalar, gcol);
          auto outputTy = cast<MemRefType>(outputMems[outputIndex].getType());
          if (!out || out.getType() != outputTy.getElementType())
            return failure();
          b.create<memref::StoreOp>(loc, out, outputMems[outputIndex],
                                    ValueRange{grow, gcol});
        }
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    SmallVector<Attribute> trace{
        StringAttr::get(ctx, "cuda.contraction.mma-sync.v1")};
    if (warpsN > 1)
      trace.push_back(StringAttr::get(
          ctx, "cuda.contraction.mma-sync.warp-n-partition.v1"));
    if (hasRegisterBiasSavedEpilogue)
      trace.push_back(StringAttr::get(
          ctx, "cuda.contraction.packed_bias_dual_output.v1"));
    auto i64 = IntegerType::get(ctx, 64);
    gpuFn->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(ctx, "contraction"));
    gpuFn->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(ctx, "single_stage_mma_sync"));
    gpuFn->setAttr("intent.backend.approximation_mode",
                   StringAttr::get(ctx, "none"));
    gpuFn->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(ctx, trace));
    gpuFn->setAttr("intent.backend.warps_n",
                   IntegerAttr::get(i64, warpsN));
    gpuFn->setAttr("intent.backend.group_m",
                   IntegerAttr::get(i64, groupM));
    gpuFn->setAttr("intent.backend.stage_count", IntegerAttr::get(i64, 1));
    gpuFn->setAttr("intent.backend.reduction_parts", IntegerAttr::get(i64, 1));
    int64_t barrierCount = 0;
    gpuFn.walk([&](gpu::BarrierOp) { ++barrierCount; });
    gpuFn->setAttr("intent.backend.fma_count", IntegerAttr::get(i64, 0));
    gpuFn->setAttr("intent.backend.shuffle_count", IntegerAttr::get(i64, 0));
    gpuFn->setAttr("intent.backend.barrier_count",
                   IntegerAttr::get(i64, barrierCount));
    gpuFn->setAttr("intent.backend.all_reduce_count", IntegerAttr::get(i64, 0));
    gpuFn->setAttr("intent.backend.block_threads",
                   IntegerAttr::get(i64, blockThreads));
    gpuFn->setAttr("intent.backend.grid_blocks",
                   IntegerAttr::get(i64, gridM * gridN));
    gpuFn->setAttr("intent.backend.predicated_value_if_count",
                   IntegerAttr::get(i64, 0));
    gpuFn->setAttr("intent.backend.folded_constexpr_select_count",
                   IntegerAttr::get(i64, 0));
    gpuFn->setAttr("intent.backend.vector_width",
                   IntegerAttr::get(i64,
                                    hasRegisterBiasSavedEpilogue ? 2 : 1));

    fn.erase();
    return success();
  }
};

struct ConvertIntentContractionToGPUMMAPass
    : ::intent::impl::ConvertIntentContractionToGPUMMABase<
          ConvertIntentContractionToGPUMMAPass> {
  using ConvertIntentContractionToGPUMMABase::
      ConvertIntentContractionToGPUMMABase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ContractionMMALowering probe(fn, (int)numStages, (int64_t)blockM,
                                   (int64_t)blockN, (int64_t)blockK,
                                   (int64_t)smemPadK,
                                   (int64_t)warpsN,
                                   (int64_t)groupM,
                                   (int64_t)splitKParts);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ContractionMMALowering low(fn, (int)numStages, (int64_t)blockM,
                                 (int64_t)blockN, (int64_t)blockK,
                                 (int64_t)smemPadK,
                                 (int64_t)warpsN,
                                 (int64_t)groupM,
                                 (int64_t)splitKParts);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("contraction-mma lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
