//===- ConvertIntentConvToGPUMMA.cpp - conv2d -> implicit-GEMM tensor core ==//
//
// The tensor-core sibling of ConvertIntentConvToGPU (the naive direct windowed
// f32-accumulation path). Lowers a func whose body is exactly ONE forward
// `intent.conv` conv2d (R=2, groups=1, static f32) as an IMPLICIT GEMM onto the
// nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16) tensor-core contraction:
//
//   Y[N,Cout,OH,OW] = im2col(X)[M, K] . W[Cout, K]^T
//     M = N*OH*OW  (output pixels, GEMM rows)
//     N_gemm = Cout                              (GEMM cols)
//     K = Cin*kH*kW                              (contracted)
//
// The im2col matrix is NEVER materialized. Each (m-row, k-col) of the A smem
// tile is mapped BACK to an input pixel by the EXACT OOB-guarded conv address
// math borrowed from ConvertIntentConvToGPU (ih = oh*sH - pH + kh*dH, etc.),
// and loaded (or zero for out-of-bounds pad) directly into the padded f16 shared
// tile -- the address math IS the tile-fill load predicate. W is already a real
// [Cout, Cin*kH*kW] matrix in memory (row-major NCHW weight), so its tile-fill
// is a plain [N_gemm, K] gather. Both operands are cast f32 -> f16 during the
// fill (fp16 input mode: reduced precision, rel ~1e-2 vs fp32 torch, matching
// the fact that the cuDNN reference itself runs TF32). The register-level
// contraction (ldmatrix + mma.sync + per-warp vector<2x2xf32> accumulators
// carried across the K loop as scf.for iter_args) and the padded-smem tile
// geometry are reused verbatim from ConvertIntentContractionToGPUMMA; the store
// scatters each accumulator element back to the NCHW output (row -> (n,oh,ow),
// col -> co) with an optional bias add.
//
// Single-buffered staging (fill -> barrier -> mma -> barrier): the im2col gather
// is not contiguous so cp.async (which needs a contiguous source) is not used
// for A; the win is the tensor-core inner loop + tile reuse vs the naive path's
// scalar per-cell accumulation at ~6% of f32 peak.
//
// Static shapes, groups==1, K % BK == 0 (else the pass declines and the naive
// -convert-intent-conv-to-gpu fallback handles it). stride/padding/dilation are
// fully general (folded into the address math). Bare-ptr ABI; scf_to_cf=True.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

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
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONVTOGPUMMA
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// mma.sync m16n8k16 tile geometry (identical to the contraction pass).
static constexpr int64_t BK = 32; // K staging step (= 2 k16 mma sub-tiles)
static constexpr int64_t MMA_M = 16, MMA_N = 8, MMA_K = 16;

struct ConvMMALowering {
  func::FuncOp fn;
  int64_t reqBM, reqBN;

  int64_t BM = 64, BN = 64;
  ConvOp conv;
  RankedTensorType inTy, wTy, outTy;
  int64_t N = 0, Cin = 0, Cout = 0;
  int64_t H = 0, W = 0, kH = 0, kW = 0, OH = 0, OW = 0;
  int64_t sH = 0, sW = 0, pH = 0, pW = 0, dH = 0, dW = 0;
  int64_t Mg = 0, Ng = 0, Kg = 0; // implicit-GEMM dims
  bool hasBias = false;
  Type f16, f32;

  ConvMMALowering(func::FuncOp fn, int64_t reqBM, int64_t reqBN)
      : fn(fn), reqBM(reqBM), reqBN(reqBN) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }
  static int64_t attrAt(ArrayAttr a, int64_t i) {
    return cast<IntegerAttr>(a[i]).getInt();
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    if (!dyn_cast<func::ReturnOp>(body.getTerminator()))
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    conv = dyn_cast_or_null<ConvOp>(only);
    if (!conv)
      return failure();
    inTy = cast<RankedTensorType>(conv.getInput().getType());
    wTy = cast<RankedTensorType>(conv.getWeight().getType());
    outTy = cast<RankedTensorType>(conv.getOutput().getType());
    // conv2d only: rank 4 (N,C,H,W).
    if (inTy.getRank() != 4 || wTy.getRank() != 4 || outTy.getRank() != 4)
      return failure();
    if (!inTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !inTy.getElementType().isF32())
      return failure();
    if (conv.getGroups() != 1)
      return failure();
    N = inTy.getDimSize(0);
    Cin = inTy.getDimSize(1);
    H = inTy.getDimSize(2);
    W = inTy.getDimSize(3);
    Cout = wTy.getDimSize(0);
    if (wTy.getDimSize(1) != Cin)
      return failure();
    kH = wTy.getDimSize(2);
    kW = wTy.getDimSize(3);
    OH = outTy.getDimSize(2);
    OW = outTy.getDimSize(3);
    sH = attrAt(conv.getStride(), 0);
    sW = attrAt(conv.getStride(), 1);
    pH = attrAt(conv.getPadding(), 0);
    pW = attrAt(conv.getPadding(), 1);
    dH = attrAt(conv.getDilation(), 0);
    dW = attrAt(conv.getDilation(), 1);
    hasBias = conv.getBias() != nullptr;

    Mg = N * OH * OW;
    Ng = Cout;
    Kg = Cin * kH * kW;
    // Tile geometry: fp16 mma.sync. K must tile evenly (else fall back to the
    // naive pass). M/N are padded via ceil-grid + guarded store, so they need
    // not divide.
    BM = reqBM > 0 ? reqBM : 64;
    BN = reqBN > 0 ? reqBN : 64;
    if (Ng % BN != 0 && BN > 64)
      BN = 64;
    if (BM % MMA_M != 0 || BN % MMA_N != 0)
      return failure();
    int64_t numWarps = BM / MMA_M;
    int64_t kBlock = numWarps * 32;
    if ((BM * BK) % kBlock != 0 || (BN * BK) % kBlock != 0)
      return failure();
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    f16 = b.getF16Type();
    f32 = b.getF32Type();

    // New signature: (input, weight[, bias]) ++ [output] memrefs.
    SmallVector<Type> inTys;
    inTys.push_back(memrefOf(inTy));
    inTys.push_back(memrefOf(wTy));
    if (hasBias)
      inTys.push_back(
          memrefOf(cast<RankedTensorType>(conv.getBias().getType())));
    inTys.push_back(memrefOf(outTy));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    unsigned ai = 0;
    Value inMem = entry->getArgument(ai++);
    Value wMem = entry->getArgument(ai++);
    Value biasMem = hasBias ? entry->getArgument(ai++) : nullptr;
    Value outMem = entry->getArgument(ai++);

    int64_t numWarps = BM / MMA_M;
    int64_t kBlock = numWarps * 32;
    int64_t nFrag = BN / MMA_N;
    int64_t gridM = (Mg + BM - 1) / BM;
    int64_t gridN = (Ng + BN - 1) / BN;
    int64_t numK = (Kg + BK - 1) / BK; // K padded via guarded fill

    int64_t aBytes = BM * BK * 2, wBytes = BN * BK * 2;
    Value shBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr((int32_t)(aBytes + wBytes)));
    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM * gridN);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    auto cidx = [&](int64_t v) {
      return b.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Type i64 = b.getI64Type();
    auto ci64 = [&](int64_t v) {
      return b.create<arith::ConstantIntOp>(loc, v, 64).getResult();
    };

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = cidx(0);
    Value c2 = cidx(2), c4 = cidx(4), c8 = cidx(8), c16 = cidx(16),
          c32 = cidx(32);
    Value cBK = cidx(BK);

    // pid -> (pidM, pidN); rowBlk / colBlk the block's C-tile origin.
    Value cGridN = cidx(gridN);
    Value pidM = b.create<arith::DivUIOp>(loc, bx, cGridN);
    Value pidN = b.create<arith::RemUIOp>(loc, bx, cGridN);
    Value rowBlk = b.create<arith::MulIOp>(loc, pidM, cidx(BM));
    Value colBlk = b.create<arith::MulIOp>(loc, pidN, cidx(BN));

    // lane/warp decomposition + ldmatrix/store lane offsets (contraction pass).
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

    // Single-buffered padded f16 shared tiles.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto aShTy = MemRefType::get({BM, BK}, f16, nullptr, wgSpace);
    auto wShTy = MemRefType::get({BN, BK}, f16, nullptr, wgSpace);
    Value aSh = b.create<memref::ViewOp>(loc, aShTy, raw, c0, ValueRange{});
    Value wOff = cidx(aBytes);
    Value wSh = b.create<memref::ViewOp>(loc, wShTy, raw, wOff, ValueRange{});

    auto aFragTy = VectorType::get({4, 2}, f16);
    auto wFragTy = VectorType::get({2, 2}, f16);
    auto accTy = VectorType::get({2, 2}, f32);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));
    SmallVector<Value> accInit(nFrag, zeroAcc);

    Value cMg = cidx(Mg), cNg = cidx(Ng), cKg = cidx(Kg);
    Value cOW = cidx(OW), cOH = cidx(OH);
    Value ckW = cidx(kW), ckH = cidx(kH);
    Value zero64 = ci64(0);
    Value cH64 = ci64(H), cW64 = ci64(W);

    int64_t perA = (BM * BK) / kBlock;
    int64_t perW = (BN * BK) / kBlock;

    // Fill the A (im2col) and W tiles for K-block `kt` into f16 shared.
    auto fillTiles = [&](OpBuilder &nb, Location nl, Value kt) {
      // ---- A tile: aSh[r,c] = f16( im2col row (rowBlk+r), col (kt+c) ) ------ //
      for (int64_t it = 0; it < perA; ++it) {
        Value idx = nb.create<arith::AddIOp>(nl, tx, cidx(it * kBlock));
        Value r = nb.create<arith::DivUIOp>(nl, idx, cBK);
        Value c = nb.create<arith::RemUIOp>(nl, idx, cBK);
        Value gm = nb.create<arith::AddIOp>(nl, rowBlk, r);
        Value gk = nb.create<arith::AddIOp>(nl, kt, c);
        // decode gm -> (n, oh, ow)
        Value ow = nb.create<arith::RemUIOp>(nl, gm, cOW);
        Value tm = nb.create<arith::DivUIOp>(nl, gm, cOW);
        Value oh = nb.create<arith::RemUIOp>(nl, tm, cOH);
        Value n = nb.create<arith::DivUIOp>(nl, tm, cOH);
        // decode gk -> (ci, kh, kw)
        Value kw = nb.create<arith::RemUIOp>(nl, gk, ckW);
        Value tk = nb.create<arith::DivUIOp>(nl, gk, ckW);
        Value kh = nb.create<arith::RemUIOp>(nl, tk, ckH);
        Value ci = nb.create<arith::DivUIOp>(nl, tk, ckH);
        // ih = oh*sH - pH + kh*dH ; iw = ow*sW - pW + kw*dW  (i64, may be < 0)
        Value oh64 = nb.create<arith::IndexCastOp>(nl, i64, oh);
        Value kh64 = nb.create<arith::IndexCastOp>(nl, i64, kh);
        Value ow64 = nb.create<arith::IndexCastOp>(nl, i64, ow);
        Value kw64 = nb.create<arith::IndexCastOp>(nl, i64, kw);
        Value ih = nb.create<arith::AddIOp>(
            nl,
            nb.create<arith::SubIOp>(
                nl, nb.create<arith::MulIOp>(nl, oh64, ci64(sH)), ci64(pH)),
            nb.create<arith::MulIOp>(nl, kh64, ci64(dH)));
        Value iw = nb.create<arith::AddIOp>(
            nl,
            nb.create<arith::SubIOp>(
                nl, nb.create<arith::MulIOp>(nl, ow64, ci64(sW)), ci64(pW)),
            nb.create<arith::MulIOp>(nl, kw64, ci64(dW)));
        Value ihGe = nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::sge, ih,
                                              zero64);
        Value ihLt =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::slt, ih, cH64);
        Value iwGe = nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::sge, iw,
                                              zero64);
        Value iwLt =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::slt, iw, cW64);
        Value mIn =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ult, gm, cMg);
        Value kIn =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ult, gk, cKg);
        Value cond = nb.create<arith::AndIOp>(
            nl, nb.create<arith::AndIOp>(nl, ihGe, ihLt),
            nb.create<arith::AndIOp>(nl, iwGe, iwLt));
        cond = nb.create<arith::AndIOp>(
            nl, cond, nb.create<arith::AndIOp>(nl, mIn, kIn));
        auto ifOp = nb.create<scf::IfOp>(nl, TypeRange{f16}, cond, true);
        {
          OpBuilder tb = ifOp.getThenBodyBuilder();
          Value ihI = tb.create<arith::IndexCastOp>(nl, tb.getIndexType(), ih);
          Value iwI = tb.create<arith::IndexCastOp>(nl, tb.getIndexType(), iw);
          Value v = tb.create<memref::LoadOp>(nl, inMem,
                                              ValueRange{n, ci, ihI, iwI});
          Value vh = tb.create<arith::TruncFOp>(nl, f16, v);
          tb.create<scf::YieldOp>(nl, ValueRange{vh});
        }
        {
          OpBuilder eb = ifOp.getElseBodyBuilder();
          Value z = eb.create<arith::ConstantOp>(nl, eb.getFloatAttr(f16, 0.0));
          eb.create<scf::YieldOp>(nl, ValueRange{z});
        }
        nb.create<memref::StoreOp>(nl, ifOp.getResult(0), aSh,
                                   ValueRange{r, c});
      }
      // ---- W tile: wSh[r,c] = f16( W[colBlk+r (=co), ci, kh, kw] ) --------- //
      for (int64_t it = 0; it < perW; ++it) {
        Value idx = nb.create<arith::AddIOp>(nl, tx, cidx(it * kBlock));
        Value r = nb.create<arith::DivUIOp>(nl, idx, cBK);
        Value c = nb.create<arith::RemUIOp>(nl, idx, cBK);
        Value gn = nb.create<arith::AddIOp>(nl, colBlk, r);
        Value gk = nb.create<arith::AddIOp>(nl, kt, c);
        Value kw = nb.create<arith::RemUIOp>(nl, gk, ckW);
        Value tk = nb.create<arith::DivUIOp>(nl, gk, ckW);
        Value kh = nb.create<arith::RemUIOp>(nl, tk, ckH);
        Value ci = nb.create<arith::DivUIOp>(nl, tk, ckH);
        Value nIn =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ult, gn, cNg);
        Value kIn =
            nb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ult, gk, cKg);
        nIn = nb.create<arith::AndIOp>(nl, nIn, kIn);
        auto ifOp = nb.create<scf::IfOp>(nl, TypeRange{f16}, nIn, true);
        {
          OpBuilder tb = ifOp.getThenBodyBuilder();
          Value v = tb.create<memref::LoadOp>(nl, wMem,
                                              ValueRange{gn, ci, kh, kw});
          Value vh = tb.create<arith::TruncFOp>(nl, f16, v);
          tb.create<scf::YieldOp>(nl, ValueRange{vh});
        }
        {
          OpBuilder eb = ifOp.getElseBodyBuilder();
          Value z = eb.create<arith::ConstantOp>(nl, eb.getFloatAttr(f16, 0.0));
          eb.create<scf::YieldOp>(nl, ValueRange{z});
        }
        nb.create<memref::StoreOp>(nl, ifOp.getResult(0), wSh,
                                   ValueRange{r, c});
      }
    };

    // acc[nt] += A * W_nt over the BK/16 k-subtiles (contraction pass recipe).
    auto computeTile = [&](OpBuilder &nb, Location nl,
                           ValueRange acc) -> SmallVector<Value> {
      SmallVector<Value> cur(acc.begin(), acc.end());
      for (int64_t ks = 0; ks < BK / MMA_K; ++ks) {
        Value ksc = cidx(ks * MMA_K);
        Value aKcol = nb.create<arith::AddIOp>(nl, ksc, aColLane);
        Value aFrag = nb.create<nvgpu::LdMatrixOp>(
            nl, aFragTy, aSh, ValueRange{aRow, aKcol}, /*transpose=*/false,
            /*numTiles=*/4);
        Value wKcol = nb.create<arith::AddIOp>(nl, ksc, wColLane);
        for (int j = 0; j < nFrag; ++j) {
          Value wRow = nb.create<arith::AddIOp>(nl, cidx(j * MMA_N), laneMod8);
          Value wFrag = nb.create<nvgpu::LdMatrixOp>(
              nl, wFragTy, wSh, ValueRange{wRow, wKcol}, /*transpose=*/false,
              /*numTiles=*/2);
          cur[j] =
              nb.create<nvgpu::MmaSyncOp>(nl, aFrag, wFrag, cur[j], mmaShape);
        }
      }
      return cur;
    };

    // K loop: fill -> barrier -> mma -> barrier, accumulators as iter_args.
    Value cNumK = cidx(numK);
    auto forOp = b.create<scf::ForOp>(
        loc, c0, cNumK, c1, accInit,
        [&](OpBuilder &nb, Location nl, Value ki, ValueRange acc) {
          Value kt = nb.create<arith::MulIOp>(nl, ki, cBK);
          fillTiles(nb, nl, kt);
          nb.create<gpu::BarrierOp>(nl);
          SmallVector<Value> cur = computeTile(nb, nl, acc);
          nb.create<gpu::BarrierOp>(nl);
          nb.create<scf::YieldOp>(nl, cur);
        });
    SmallVector<Value> finalAcc(forOp.getResults().begin(),
                                forOp.getResults().end());

    // Store: d[i][j] -> row = rowBlk + warp*16 + lane/4 + i*8 (= GEMM m-row),
    //                   col = colBlk + nt*8 + (lane%4)*2 + j (= co). Guarded,
    // + optional bias, scattered to out[n, co, oh, ow] (row -> (n,oh,ow)).
    Value warpRowGlobal = b.create<arith::AddIOp>(loc, rowBlk, warpRowBase);
    for (int j = 0; j < nFrag; ++j) {
      Value colBase = b.create<arith::AddIOp>(loc, colBlk, cidx(j * MMA_N));
      for (int i = 0; i < 2; ++i) {
        Value row = b.create<arith::AddIOp>(
            loc, warpRowGlobal,
            b.create<arith::AddIOp>(loc, laneDiv4, cidx(i * 8)));
        for (int jj = 0; jj < 2; ++jj) {
          Value col = b.create<arith::AddIOp>(
              loc, colBase, b.create<arith::AddIOp>(loc, stCol, cidx(jj)));
          Value rowIn =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, cMg);
          Value colIn =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, cNg);
          Value inB = b.create<arith::AndIOp>(loc, rowIn, colIn);
          auto ifOp = b.create<scf::IfOp>(loc, inB, /*withElse=*/false);
          OpBuilder tb = ifOp.getThenBodyBuilder();
          Value v = tb.create<vector::ExtractOp>(
              loc, finalAcc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
          if (hasBias) {
            Value bv = tb.create<memref::LoadOp>(loc, biasMem, ValueRange{col});
            v = tb.create<arith::AddFOp>(loc, v, bv);
          }
          // row -> (n, oh, ow)
          Value ow = tb.create<arith::RemUIOp>(loc, row, cOW);
          Value tm = tb.create<arith::DivUIOp>(loc, row, cOW);
          Value oh = tb.create<arith::RemUIOp>(loc, tm, cOH);
          Value n = tb.create<arith::DivUIOp>(loc, tm, cOH);
          tb.create<memref::StoreOp>(loc, v, outMem,
                                     ValueRange{n, col, oh, ow});
        }
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentConvToGPUMMAPass
    : ::intent::impl::ConvertIntentConvToGPUMMABase<
          ConvertIntentConvToGPUMMAPass> {
  using ConvertIntentConvToGPUMMABase::ConvertIntentConvToGPUMMABase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ConvMMALowering probe(fn, (int64_t)blockM, (int64_t)blockN);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ConvMMALowering low(fn, (int64_t)blockM, (int64_t)blockN);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("conv-mma lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
