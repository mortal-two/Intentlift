//===- ConvertIntentContractionToGPUMMAWS.cpp - warp-specialized matmul ====//
//
// M7g: the WARP-SPECIALIZED (producer/consumer) sibling of
// ConvertIntentContractionToGPUMMA. Same recognizer and the same register-level
// nvgpu.ldmatrix + nvgpu.mma.sync contraction, but the block's warps are split
// into roles:
//
//   * PRODUCER warps (the HIGH warp ids, numConsWarps..total-1): loop over the
//     K-tiles issuing cp.async loads of A/W into an N-stage shared buffer; after
//     a tile lands (nvgpu.device_async_wait) they signal a per-stage `full`
//     mbarrier and wait the matching `empty` mbarrier before reusing a buffer.
//   * CONSUMER warps (the LOW warp ids 0..numConsWarps-1, each owning a 16-row C
//     strip exactly as the M7f layout): loop over the K-tiles waiting the per-
//     stage `full` mbarrier, running ldmatrix + mma.sync into the register
//     accumulator (scf.for iter_args), then signaling the `empty` mbarrier.
//
// The producer/consumer handshake uses the nvgpu.mbarrier primitives (G5-probed
// to lower to mbarrier.{init,arrive,try_wait.parity}.shared.b64 on sm_120a and
// to execute correctly: a producer warp's writes are observed by a decoupled
// consumer warp). Phase parity for `try_wait.parity` is computed directly from
// the loop index -- stage s = ki%nStages is on its (ki/nStages)-th use, whose
// parity is (ki/nStages)%2; this holds for BOTH the consumer's `full` wait and
// the producer's `empty` wait (the latter pre-primed by the consumers arriving
// `empty` once per stage before the K-loop). No phase iter_args needed.
//
// Decoupling the load-issuing warps from the mma warps is the schedule that
// tl.dot/cuBLAS use to hide load latency behind a deep async pipeline; this pass
// is the experiment for whether that closes the residual mma.sync gap on
// sm_120a. The M7f pass is left 100% intact; this is gated by its own name.
//
// Static shapes, M%BM==N%BN==K%BK==0. Bare-ptr ABI; scf_to_cf=True.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "MapEmit.h"
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
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONTRACTIONTOGPUMMAWS
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static constexpr int64_t BK = 32; // K staging step (= 2 k16 mma sub-tiles)
static constexpr int64_t MMA_M = 16, MMA_N = 8, MMA_K = 16;
static constexpr int64_t GROUP_M = 8;

struct ContractionWSLowering {
  func::FuncOp fn;
  int numStages;
  int64_t reqBM, reqBN;
  int numProdWarps;

  int64_t BM = 64, BN = 64;
  int64_t M = 0, N = 0, K = 0;
  unsigned aArgIdx = 0, wArgIdx = 0;
  ContractionOp con;
  Type f16, f32;
  Value retVal;
  bool hasEpilogue = false;
  SmallVector<Role> argRoles;
  DenseMap<Value, Value> argMemref;

  ContractionWSLowering(func::FuncOp fn, int numStages, int64_t reqBM,
                        int64_t reqBN, int numProdWarps)
      : fn(fn), numStages(numStages), reqBM(reqBM), reqBN(reqBN),
        numProdWarps(numProdWarps) {}

  // ---- recognition (identical to the M7f MMA pass) ----------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 1)
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
      } else if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, arith::ConstantOp>(
                     &op)) {
        return failure();
      }
    }
    if (!c)
      return failure();
    con = c;
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    retVal = ret.getOperand(0);
    hasEpilogue = retVal != c.getResult();
    if (c.getSubscripts() != "mk,nk->mn")
      return failure();

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

    BM = reqBM;
    BN = reqBN;
    if (M % BM != 0 || N % BN != 0) {
      BM = 64;
      BN = 64;
    }
    if (K % BK != 0 || M % BM != 0 || N % BN != 0)
      return failure();
    if (BM % MMA_M != 0 || BN % MMA_N != 0)
      return failure();

    auto inputs = fty.getInputs();
    argRoles.assign(inputs.size(), Role::Row);
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (i == aArgIdx || i == wArgIdx) {
        argRoles[i] = Role::Row;
        continue;
      }
      auto t = dyn_cast<RankedTensorType>(inputs[i]);
      if (!t || !t.hasStaticShape() || !isa<Float32Type>(t.getElementType()))
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

  // ---- epilogue scalar evaluator (identical to the M7f MMA pass) --------- //
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

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

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

    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);

    int64_t numConsWarps = BM / MMA_M;
    int64_t totalWarps = numConsWarps + numProdWarps;
    int64_t blockThreads = totalWarps * 32;
    int64_t consThreads = numConsWarps * 32;
    int64_t prodThreads = numProdWarps * 32;
    int64_t nFrag = BN / MMA_N;

    int64_t gridN = N / BN, gridM = M / BM;
    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM * gridN);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, blockThreads);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    int64_t numK = K / BK;
    int nStages = std::min<int64_t>((int64_t)numStages, numK);
    if (nStages < 1)
      nStages = 1;

    int64_t aBytes = BM * BK * 2, wBytes = BN * BK * 2;
    int64_t aAllBytes = nStages * aBytes, wAllBytes = nStages * wBytes;
    int64_t cBytes = hasEpilogue ? BM * BN * 4 : 0;
    Value shBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr((int32_t)(aAllBytes + wAllBytes + cBytes)));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cBK = b.create<arith::ConstantIndexOp>(loc, BK);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value ticks = b.create<arith::ConstantIndexOp>(loc, 10000000);

    // Flat pid -> (pid_m, pid_n) with the GROUP_M L2 swizzle.
    Value cGroupM = b.create<arith::ConstantIndexOp>(loc, GROUP_M);
    Value cGridM = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cWidth = b.create<arith::ConstantIndexOp>(loc, GROUP_M * gridN);
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

    // lane/warp decomposition + ldmatrix/store lane offsets (consumer layout).
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = b.create<arith::RemUIOp>(loc, tx, c32);
    Value warpRowBase = b.create<arith::MulIOp>(loc, warp, c16);
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);
    Value laneDiv16 = b.create<arith::DivUIOp>(loc, lane, c16);
    Value aColLane = b.create<arith::MulIOp>(loc, laneDiv16, c8);
    Value aRow = b.create<arith::AddIOp>(loc, warpRowBase, laneMod16);
    Value laneMod8 = b.create<arith::RemUIOp>(loc, lane, c8);
    Value laneDiv8 = b.create<arith::DivUIOp>(loc, lane, c8);
    Value wColLane = b.create<arith::MulIOp>(loc, laneDiv8, c8);
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value stCol = b.create<arith::MulIOp>(loc, laneMod4, c2);

    // Producer-local tid (valid in producer branch): tx - consThreads.
    Value cConsThreads = b.create<arith::ConstantIndexOp>(loc, consThreads);
    Value prodTid = b.create<arith::SubIOp>(loc, tx, cConsThreads);

    // Dynamic shared staging (rotating N-stage buffers).
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto aShTy = MemRefType::get({nStages, BM, BK}, f16, nullptr, wgSpace);
    auto wShTy = MemRefType::get({nStages, BN, BK}, f16, nullptr, wgSpace);
    Value aSh = b.create<memref::ViewOp>(loc, aShTy, raw, c0, ValueRange{});
    Value wOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes);
    Value wSh = b.create<memref::ViewOp>(loc, wShTy, raw, wOff, ValueRange{});
    Value cSh;
    if (hasEpilogue) {
      auto cShTy = MemRefType::get({BM, BN}, f32, nullptr, wgSpace);
      Value cOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes + wAllBytes);
      cSh = b.create<memref::ViewOp>(loc, cShTy, raw, cOff, ValueRange{});
    }

    // mbarrier groups: full[s] (producer->consumer), empty[s] (consumer->prod).
    auto barTy = nvgpu::MBarrierGroupType::get(ctx, wgSpace, nStages);
    Value fullBar = b.create<nvgpu::MBarrierCreateOp>(loc, barTy);
    Value emptyBar = b.create<nvgpu::MBarrierCreateOp>(loc, barTy);
    Value cProdThreads = b.create<arith::ConstantIndexOp>(loc, prodThreads);
    Value cConsThreads2 = cConsThreads;
    Value isT0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, c0);
    b.create<scf::IfOp>(loc, isT0, [&](OpBuilder &nb, Location nl) {
      for (int s = 0; s < nStages; ++s) {
        Value sc = nb.create<arith::ConstantIndexOp>(nl, s);
        nb.create<nvgpu::MBarrierInitOp>(nl, fullBar, cProdThreads, sc, Value());
        nb.create<nvgpu::MBarrierInitOp>(nl, emptyBar, cConsThreads2, sc,
                                         Value());
      }
      nb.create<scf::YieldOp>(nl);
    });
    b.create<gpu::BarrierOp>(loc);

    auto accTy = VectorType::get({2, 2}, f32);
    auto aFragTy = VectorType::get({4, 2}, f16);
    auto wFragTy = VectorType::get({2, 2}, f16);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));
    SmallVector<Value> accInit(nFrag, zeroAcc);

    auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    auto barTokTy = nvgpu::MBarrierTokenType::get(ctx);
    IntegerAttr dstElems = b.getIndexAttr(8);
    UnitAttr bypassL1 = b.getUnitAttr();
    int64_t colsPerRow = BK / 8;
    int64_t vPerThreadA = (BM * BK) / 8 / prodThreads;
    int64_t vPerThreadW = (BN * BK) / 8 / prodThreads;
    Value cCPR = b.create<arith::ConstantIndexOp>(loc, colsPerRow);
    Value cNStages = b.create<arith::ConstantIndexOp>(loc, nStages);
    Value c2idx = c2;

    // Consumer `full` wait parity: stage s=ki%nStages is on its (ki/nStages)-th
    // use; the producer's (ki/nStages)-th `full` arrival completes that phase
    // with parity (ki/nStages)%2.
    auto fullParity = [&](OpBuilder &nb, Location nl, Value ki) -> Value {
      Value q = nb.create<arith::DivUIOp>(nl, ki, cNStages);
      Value p = nb.create<arith::RemUIOp>(nl, q, c2idx);
      return nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ne, p, c0);
    };
    // Producer `empty` wait parity (only for ki >= nStages, i.e. the 2nd+ use of
    // a buffer): the buffer was last freed by the consumer's (m-1)-th `empty`
    // arrival, completing parity (m-1)%2, where m = ki/nStages. No pre-prime:
    // the producer simply skips the wait on a buffer's first use.
    auto emptyParity = [&](OpBuilder &nb, Location nl, Value ki) -> Value {
      Value q = nb.create<arith::DivUIOp>(nl, ki, cNStages);
      Value qm1 = nb.create<arith::SubIOp>(nl, q, c1);
      Value p = nb.create<arith::RemUIOp>(nl, qm1, c2idx);
      return nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ne, p, c0);
    };

    // Producer: cp.async load tile `ktVal` into buffer `stageVal`.
    auto loadTile = [&](OpBuilder &nb, Location nl, Value ktVal,
                        Value stageVal) -> Value {
      SmallVector<Value> toks;
      for (int64_t it = 0; it < vPerThreadA; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * prodThreads);
        Value vt = nb.create<arith::AddIOp>(nl, prodTid, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8);
        Value gk = nb.create<arith::AddIOp>(nl, ktVal, cc);
        Value ar = nb.create<arith::AddIOp>(nl, rowBlk, r);
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, aSh, ValueRange{stageVal, r, cc}, aMem,
            ValueRange{ar, gk}, dstElems, /*srcElements=*/Value(), bypassL1));
      }
      for (int64_t it = 0; it < vPerThreadW; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * prodThreads);
        Value vt = nb.create<arith::AddIOp>(nl, prodTid, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8);
        Value gk = nb.create<arith::AddIOp>(nl, ktVal, cc);
        Value wr = nb.create<arith::AddIOp>(nl, colBlk, r);
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, wSh, ValueRange{stageVal, r, cc}, wMem,
            ValueRange{wr, gk}, dstElems, /*srcElements=*/Value(), bypassL1));
      }
      return nb.create<nvgpu::DeviceAsyncCreateGroupOp>(nl, tokTy, toks);
    };

    // Consumer: acc[nt] += A(stage) * W_nt(stage) over the BK/16 k-subtiles.
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
          Value nbase = nb.create<arith::ConstantIndexOp>(nl, j * MMA_N);
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
    Value cNumConsWarps =
        b.create<arith::ConstantIndexOp>(loc, numConsWarps);
    Value isConsumer = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, warp, cNumConsWarps);

    // ===================== producer / consumer split ===================== //
    b.create<scf::IfOp>(
        loc, isConsumer,
        /*thenBuilder (CONSUMER)=*/
        [&](OpBuilder &nb, Location nl) {
          Value cNumK = nb.create<arith::ConstantIndexOp>(nl, numK);
          auto forOp = nb.create<scf::ForOp>(
              nl, c0, cNumK, c1, accInit,
              [&](OpBuilder &fb, Location fl, Value ki, ValueRange acc) {
                Value s = fb.create<arith::RemUIOp>(fl, ki, cNStages);
                Value par = fullParity(fb, fl, ki);
                fb.create<nvgpu::MBarrierTryWaitParityOp>(fl, fullBar, par,
                                                          ticks, s);
                SmallVector<Value> cur = computeTile(fb, fl, s, acc);
                fb.create<nvgpu::MBarrierArriveOp>(fl, barTokTy, emptyBar, s);
                fb.create<scf::YieldOp>(fl, cur);
              });
          SmallVector<Value> finalAcc(forOp.getResults().begin(),
                                      forOp.getResults().end());
          if (!hasEpilogue) {
            for (int j = 0; j < nFrag; ++j) {
              Value nbase = nb.create<arith::ConstantIndexOp>(nl, j * MMA_N);
              Value colBase = nb.create<arith::AddIOp>(nl, colBlk, nbase);
              storeAcc(nb, nl, finalAcc[j], warpRowGlobal, colBase,
                       [&](OpBuilder &sb, Location sl, Value v, Value row,
                           Value col) {
                         sb.create<memref::StoreOp>(sl, v, yMem,
                                                    ValueRange{row, col});
                       });
            }
          } else {
            // spill consumer accumulators to the shared [BM,BN] C tile.
            for (int j = 0; j < nFrag; ++j) {
              Value nbase = nb.create<arith::ConstantIndexOp>(nl, j * MMA_N);
              storeAcc(nb, nl, finalAcc[j], warpRowBase, nbase,
                       [&](OpBuilder &sb, Location sl, Value v, Value row,
                           Value col) {
                         sb.create<memref::StoreOp>(sl, v, cSh,
                                                    ValueRange{row, col});
                       });
            }
          }
          nb.create<scf::YieldOp>(nl);
        },
        /*elseBuilder (PRODUCER)=*/
        [&](OpBuilder &nb, Location nl) {
          Value cNumK = nb.create<arith::ConstantIndexOp>(nl, numK);
          Value cNStagesIdx = cNStages;
          nb.create<scf::ForOp>(
              nl, c0, cNumK, c1, ValueRange{},
              [&](OpBuilder &fb, Location fl, Value ki, ValueRange) {
                Value s = fb.create<arith::RemUIOp>(fl, ki, cNStages);
                // Wait for the buffer to be free, but only on its 2nd+ use
                // (ki >= nStages); the first nStages tiles fill empty buffers.
                Value reuse = fb.create<arith::CmpIOp>(
                    fl, arith::CmpIPredicate::uge, ki, cNStagesIdx);
                fb.create<scf::IfOp>(fl, reuse, [&](OpBuilder &gb, Location gl) {
                  Value par = emptyParity(gb, gl, ki);
                  gb.create<nvgpu::MBarrierTryWaitParityOp>(gl, emptyBar, par,
                                                            ticks, s);
                  gb.create<scf::YieldOp>(gl);
                });
                Value kt = fb.create<arith::MulIOp>(fl, ki, cBK);
                Value grp = loadTile(fb, fl, kt, s);
                fb.create<nvgpu::DeviceAsyncWaitOp>(fl, grp, IntegerAttr());
                fb.create<nvgpu::MBarrierArriveOp>(fl, barTokTy, fullBar, s);
                fb.create<scf::YieldOp>(fl);
              });
          nb.create<scf::YieldOp>(nl);
        });

    // Cooperative bias/activation epilogue (all threads, guarded).
    if (hasEpilogue) {
      b.create<gpu::BarrierOp>(loc);
      Value cBN = b.create<arith::ConstantIndexOp>(loc, BN);
      Value cTotal = b.create<arith::ConstantIndexOp>(loc, BM * BN);
      int64_t epIters = (BM * BN + blockThreads - 1) / blockThreads;
      for (int64_t it = 0; it < epIters; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * blockThreads);
        Value i = b.create<arith::AddIOp>(loc, tx, base);
        Value inb =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, cTotal);
        b.create<scf::IfOp>(loc, inb, [&](OpBuilder &nb, Location nl) {
          Value r = nb.create<arith::DivUIOp>(nl, i, cBN);
          Value cc = nb.create<arith::RemUIOp>(nl, i, cBN);
          Value cScalar = nb.create<memref::LoadOp>(nl, cSh, ValueRange{r, cc});
          Value gcol = nb.create<arith::AddIOp>(nl, colBlk, cc);
          Value grow = nb.create<arith::AddIOp>(nl, rowBlk, r);
          Value out = emitEpilogue(nb, nl, retVal, cScalar, gcol);
          if (out)
            nb.create<memref::StoreOp>(nl, out, yMem, ValueRange{grow, gcol});
          nb.create<scf::YieldOp>(nl);
        });
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentContractionToGPUMMAWSPass
    : ::intent::impl::ConvertIntentContractionToGPUMMAWSBase<
          ConvertIntentContractionToGPUMMAWSPass> {
  using ConvertIntentContractionToGPUMMAWSBase::
      ConvertIntentContractionToGPUMMAWSBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ContractionWSLowering probe(fn, (int)numStages, (int64_t)blockM,
                                  (int64_t)blockN, (int)numProducerWarps);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ContractionWSLowering low(fn, (int)numStages, (int64_t)blockM,
                                (int64_t)blockN, (int)numProducerWarps);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("contraction-mma-ws lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
