//===- ConvertIntentContractionToGPU.cpp - matmul -> tensor-core gpu.launch ===//
//
// Lowers a func whose body is one `intent.contraction{"mk,nk->mn"}`
// (A[M,K] f16, W[N,K] f16 -> Y[M,N] f32) -- optionally followed by a pointwise
// epilogue chain (bias-add + activation) -- to ONE tiled-K tensor-core
// `gpu.launch`. This is the first CUDA-core-free path: the contraction is
// realized by NVIDIA WMMA (`gpu.subgroup_mma_*` -> wmma.mma.sync.aligned
// .row.row.m16n16k16.f32.f32 on sm_120a), NOT scalarized to a `linalg.generic`
// MAC (that leg is left in ConvertIntentToLinalg for the CPU/oracle path).
//
// M7b fused epilogue (the linear.kernel_fwd structure activation(A@W^T+bias)):
// when the contraction result feeds a straight-line {map_unary, map_binary,
// select} chain (bias-add against a Col[N] arg + an activation expressed in the
// scalar vocab -- squared_relu = mul(select(x>=0,x,0), x), gelu_approx = the
// 0.5*x*(1+tanh(...)) tanh form), the chain is applied to the COp accumulator
// fragments BEFORE the store: each fragment is spilled to a dynamic-shared
// staging tile, then the block cooperatively reloads every element, evaluates
// the epilogue scalar DAG (reusing MapEmit.h emitScalar{Unary,Binary,Cmp} +
// arith.select), and writes Y. The id case (return == contraction) keeps the
// direct subgroup_mma_store_matrix path byte-identical to M7a.
//
// Schedule (BM=BN=64, BK=32 = 2 m16n16k16 k-subtiles; a 4x4 grid of m16n16k16 C tiles, 4 warps/block):
//   grid  = (cdiv(M,BM)*cdiv(N,BN), 1, 1)  flat pid -> (pid_m, pid_n)
//   block = (128, 1, 1)                    warp w = tx/32 owns C rows [w*16,+16)
// M7c cp.async multistage pipeline: the K loop is software-pipelined over N
// (default 3) rotating dynamic-shared buffers so the global->shared loads of
// tiles K+1..K+N-1 overlap the WMMA compute of tile K (the old scalar
// memref.load/store cooperative copy serialized staging behind compute and left
// the kernel DRAM-bound -- 4096^3 plateaued at ~17 TFLOP/s). Each A/W tile is
// streamed by nvgpu.device_async_copy (-> cp.async.cg.shared.global, 16-byte/8-
// f16 vectors, L1 bypassed), grouped by device_async_create_group, synchronized
// by device_async_wait{numGroups}. N is clamped to the K-tile count (a 1-/2-trip
// smoke shape degenerates to N=1/N=2 through the same code path).
// Per block:
//   4 fp32 COp accumulator fragments (one per the warp's 4 column tiles), init 0
//   PROLOGUE: issue cp.async for tiles 0..N-2 into buffers 0..N-2 (keep the last
//     group token live -- it dominates + is the numGroups-driven epilogue wait)
//   scf.for ki = 0..numK-N iter_args(the 4 COp fragments):  // STEADY STATE
//     prefetch tile ki+N-1 -> buffer (ki+N-1)%N (cp.async), device_async_wait
//       {numGroups=N-1} (tile ki ready, N-1 groups still in flight), gpu.barrier
//     compute tile ki from buffer ki%N: for ks in 0..BK/16 load the warp's A row
//       sub-tile (AOp); for j in 0..3 load the W column tile (BOp, transposed so
//       "nk" -> the k-major B operand) and subgroup_mma_compute COp_j += A * B_j
//     gpu.barrier (reads done before the buffer is reused N iters later); yield
//   EPILOGUE: compute the last N-1 (already-prefetched) tiles, draining one
//     in-flight group each via a decreasing static numGroups
//   store: subgroup_mma_store_matrix each COp_j to Y[w*16, j*16]
//
// The transposed B load makes "mk,nk->mn" exact: B[k,n] = W[n,k], so the MMA
// computes C[m,n] += sum_k A[m,k]*W[n,k] (verified rel ~1e-7 vs torch.matmul).
// Static shapes, M%64==N%64==K%32==0 (tail-mask is a documented follow-up).
// Bare-ptr ABI; the lowered module goes to lower_memref_module_to_kernels with
// scf_to_cf=True (the gpu->nvvm pipeline needs scf->cf BEFORE it to legalize an
// scf.for carrying !gpu.mma_matrix iter_args).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONTRACTIONTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Tensor-core tile geometry. BM/BN = the block's C tile (a pass option; default
// BM=64, BN=128 = tl.dot's winning large-shape config). BK = the K staging step
// (32 = 2 m16n16k16 sub-tiles). Warps tile the M axis: numWarps = BM/16, each
// warp owns one 16-row strip spanning all BN columns = BN/16 m16n16k16 C tiles.
// Widening BN (more column fragments per warp) raises A-load reuse: one A
// fragment feeds BN/16 MMAs (M7d: the 25-34% K-bound residual was this register/
// tile blocking, not the cp.async pipeline -- which num-stages sweeps confirm is
// saturated). kBlock = numWarps*32 is derived per config in rewrite().
static constexpr int64_t BK = 32, kWMMA = 16;
// L2-swizzle group width (matches tl.dot's GROUP_M=8): consecutive program ids
// walk a GROUP_M-tall column of C blocks so neighbours reuse the same A rows /
// W columns in L2. Decisive on large K-bound shapes (4096^3 was L2-thrashing).
static constexpr int64_t GROUP_M = 8;

/// Recognize the rank-2 row-by-row contraction independently of the author's
/// einsum letters.  The J spellings ``mk,nk->mn`` and ``ik,jk->ij`` expose the
/// same canonical WMMA implementation structure.
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

/// One contraction func's tensor-core lowering. Reads the original (tensor)
/// func body (exactly one intent.contraction), builds a new memref+gpu.launch
/// func with the same name, erases the old.
struct ContractionLowering {
  func::FuncOp fn;
  int numStages; // requested cp.async pipeline depth (clamped to the K-tile count)
  int64_t reqBM, reqBN; // requested block tile (pass option)

  // populated by match()
  int64_t BM = 64, BN = 128; // effective block tile (falls back to 64x64 for
                             // shapes the requested tile does not divide)
  int64_t M = 0, N = 0, K = 0;
  unsigned aArgIdx = 0, wArgIdx = 0; // which func args are A[M,K] / W[N,K]
  ContractionOp con;
  Type f16, f32;
  // The canonical path keeps every observable result from J.  A direct
  // contraction has one f32 result; a typed epilogue may return several f16/f32
  // results, all produced by the same launch.
  SmallVector<Value> retVals;
  bool hasEpilogue = false;
  SmallVector<Role> argRoles;       // per func arg (A/W = Row, bias = Col, ...)
  DenseMap<Value, Value> argMemref; // src block arg -> new memref arg

  ContractionLowering(func::FuncOp fn, int numStages, int64_t reqBM,
                      int64_t reqBN)
      : fn(fn), numStages(numStages), reqBM(reqBM), reqBN(reqBN) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() == 0)
      return failure();
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    // Body: exactly one intent.contraction, followed by an OPTIONAL straight-line
    // pointwise epilogue ({map_unary, map_binary, select}) -- the bias-add +
    // activation chain of linear.kernel_fwd. Anything else (a second
    // contraction, a reduce, a loop) is out of scope.
    ContractionOp c;
    for (Operation &op : body.without_terminator()) {
      if (auto cc = dyn_cast<ContractionOp>(&op)) {
        if (c)
          return failure(); // more than one contraction
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

    // Canonical live graphs spell the storage-to-accumulator conversion as an
    // explicit f16->f32 cast.  Peel exactly that lossless widening: the target
    // kernel still reads the original 16-bit ABI and accumulates in f32.
    auto peelF16Input = [&](Value operand) -> BlockArgument {
      if (auto arg = dyn_cast<BlockArgument>(operand))
        return arg;
      auto castOp = operand.getDefiningOp<CastOp>();
      if (!castOp)
        return {};
      auto srcTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
      auto dstTy = dyn_cast<RankedTensorType>(castOp.getResult().getType());
      if (!srcTy || !dstTy || srcTy.getShape() != dstTy.getShape() ||
          !isa<Float16Type>(srcTy.getElementType()) ||
          !isa<Float32Type>(dstTy.getElementType()))
        return {};
      return dyn_cast<BlockArgument>(castOp.getInput());
    };
    auto aArg = peelF16Input(c.getA());
    auto wArg = peelF16Input(c.getB());
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
    if (!isa<Float16Type>(aTy.getElementType()) ||
        !isa<Float16Type>(wTy.getElementType()) ||
        !isa<Float32Type>(yTy.getElementType()))
      return failure();
    if (conATy.getShape() != aTy.getShape() ||
        conWTy.getShape() != wTy.getShape() ||
        (conATy.getElementType() != aTy.getElementType() &&
         !isa<Float32Type>(conATy.getElementType())) ||
        (conWTy.getElementType() != wTy.getElementType() &&
         !isa<Float32Type>(conWTy.getElementType())))
      return failure();
    f16 = aTy.getElementType();
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
          (!isa<Float16Type>(resultTy.getElementType()) &&
           resultTy.getElementType() != f32))
        return failure();
    }
    // Divisibility: full tiles only (non-divisible tail-mask is a follow-up).
    // Pick the effective block tile: the requested (BM,BN) if it divides, else
    // fall back to 64x64 (covers the smoke shapes 64x64x32 / 128x128x64 that the
    // wide default BN=128 would not divide).
    BM = reqBM;
    BN = reqBN;
    if (M % BM != 0 || N % BN != 0) {
      BM = 64;
      BN = 64;
    } else if (BN > 64 && N % 64 == 0) {
      // Grid-aware narrowing: a wider BN raises per-warp compute reuse but only
      // pays off when the launch still has enough blocks to fill every SM. Below
      // ~64 blocks the wide tile under-occupies and LOSES (measured on the 5090D:
      // 512^3 at BN=128 = 32 blocks regressed mine/tl.dot 0.94 -> 1.27, while
      // 1024^2x8192 / 4096^3 with 128 / 2048 blocks each gained ~2-6%). Narrow
      // back to BN=64 when the wide grid would starve.
      int64_t wideBlocks = (M / BM) * (N / BN);
      if (wideBlocks < 64)
        BN = 64;
    }
    if (K % BK != 0 || M % BM != 0 || N % BN != 0)
      return failure();

    // Classify every func arg by role for the epilogue evaluator: A/W are the
    // contraction operands (Row, never read in the epilogue); a rank-1 [N] f32
    // is a Col bias; a rank-0 f32 is a Scalar. Anything else is out of scope.
    auto inputs = fty.getInputs();
    argRoles.assign(inputs.size(), Role::Row);
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (i == aArgIdx || i == wArgIdx) {
        argRoles[i] = Role::Row;
        continue;
      }
      auto t = dyn_cast<RankedTensorType>(inputs[i]);
      if (!t || !t.hasStaticShape() ||
          (!isa<Float16Type>(t.getElementType()) &&
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

  // ---- epilogue scalar evaluator ----------------------------------------- //
  // Evaluate an epilogue value at one output element: `cScalar` is the
  // contraction's accumulated f32 (reloaded from the staging tile), `gcol` is
  // the element's GLOBAL output column (for the Col bias gather). Mirrors the
  // elementwise pass's emitAt, reusing the shared MapEmit.h scalar emitters.
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
      return nullptr; // Row (A/W) must not appear in the epilogue
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      // Splat (or rank-0) constant tensor -> the scalar f32 constant.
      if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
        if (dense.isSplat())
          return b.create<arith::ConstantOp>(
              loc, b.getF32FloatAttr(
                       dense.getSplatValue<APFloat>().convertToDouble()));
      return nullptr;
    }
    if (auto castOp = dyn_cast<CastOp>(def)) {
      Value source = emitEpilogue(b, loc, castOp.getInput(), cScalar, gcol);
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

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: original args as memrefs ++ one destination per observable
    // J result.  Multi-output saved activations are not dropped or reconstructed
    // outside the launch.
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      auto t = cast<RankedTensorType>(at);
      inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    for (Type resultType : fn.getFunctionType().getResults()) {
      auto t = cast<RankedTensorType>(resultType);
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

    // Map the original tensor block args to the new memref args (the epilogue
    // evaluator gathers Col bias / Scalar args through this map).
    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);

    // Warp layout: numWarps warps tile the M axis (one 16-row strip each); each
    // warp owns nFrag = BN/16 column fragments. kBlock = numWarps * 32 threads.
    int64_t numWarps = BM / kWMMA;
    int64_t kBlock = numWarps * 32;
    int64_t nFrag = BN / kWMMA;

    int64_t gridN = N / BN, gridM = M / BM;
    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM * gridN);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    // cp.async multistage pipeline depth: N rotating shared buffers double/
    // triple-buffer the K loop so global->shared loads of tiles K+1..K+N-1
    // overlap the WMMA compute of tile K. Clamp N to the K-tile count (a 1- or
    // 2-trip smoke shape degenerates to N=1/N=2 through the SAME code path).
    int64_t numK = K / BK;
    int nStages = std::min<int64_t>((int64_t)numStages, numK);
    if (nStages < 1)
      nStages = 1;

    // Dynamic shared: N staged A tiles [BM,BK] f16 ++ N staged W tiles [BN,BK]
    // f16 ++ (fused-epilogue path) the [BM,BN] f32 accumulator-spill tile.
    int64_t aBytes = BM * BK * 2, wBytes = BN * BK * 2;          // one stage
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
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cWMMA = b.create<arith::ConstantIndexOp>(loc, kWMMA);

    // Flat pid -> (pid_m, pid_n) with the GROUP_M L2 swizzle (tl.dot's decode):
    //   width = GROUP_M*gridN; group_id = pid/width
    //   group_size = min(gridM - group_id*GROUP_M, GROUP_M)
    //   pid_m = group_id*GROUP_M + pid%group_size
    //   pid_n = (pid%width)/group_size
    Value cGroupM = b.create<arith::ConstantIndexOp>(loc, GROUP_M);
    Value cGridM = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cWidth = b.create<arith::ConstantIndexOp>(loc, GROUP_M * gridN);
    Value groupId = b.create<arith::DivUIOp>(loc, bx, cWidth);
    Value pidInW = b.create<arith::RemUIOp>(loc, bx, cWidth);
    Value gmStart = b.create<arith::MulIOp>(loc, groupId, cGroupM);
    Value remM = b.create<arith::SubIOp>(loc, cGridM, gmStart);
    Value groupSz = b.create<arith::MinUIOp>(loc, remM, cGroupM);
    Value pidM =
        b.create<arith::AddIOp>(loc, gmStart,
                                b.create<arith::RemUIOp>(loc, bx, groupSz));
    Value pidN = b.create<arith::DivUIOp>(loc, pidInW, groupSz);
    Value cBMv = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBNv = b.create<arith::ConstantIndexOp>(loc, BN);
    Value rowBlk = b.create<arith::MulIOp>(loc, pidM, cBMv);
    Value colBlk = b.create<arith::MulIOp>(loc, pidN, cBNv);
    // warp w = tx/32 owns C rows [rowBlk + w*16, +16).
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value warpRow = b.create<arith::MulIOp>(loc, warp, cWMMA);
    Value myRow = b.create<arith::AddIOp>(loc, rowBlk, warpRow);

    // Dynamic-shared staging buffers (the G1-proven config: dynamic_shared +
    // memref.view, not a workgroup memref.alloca). The A/W tiles are rank-3
    // [N,BM,BK] so the rotating stage is a plain index operand on the async-copy
    // dst and the WMMA load src (probed: a rank-3 shared src lowers through
    // wmma.load.{a,b}...shared.f16 -- no fragile arith.select-on-memref needed).
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto aShTy = MemRefType::get({nStages, BM, BK}, f16, nullptr, wgSpace);
    auto wShTy = MemRefType::get({nStages, BN, BK}, f16, nullptr, wgSpace);
    Value aSh = b.create<memref::ViewOp>(loc, aShTy, raw, c0, ValueRange{});
    Value wOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes);
    Value wSh = b.create<memref::ViewOp>(loc, wShTy, raw, wOff, ValueRange{});
    Value cSh; // [BM,BN] f32 accumulator-spill tile (fused-epilogue path only)
    if (hasEpilogue) {
      auto cShTy = MemRefType::get({BM, BN}, f32, nullptr, wgSpace);
      Value cOff = b.create<arith::ConstantIndexOp>(loc, aAllBytes + wAllBytes);
      cSh = b.create<memref::ViewOp>(loc, cShTy, raw, cOff, ValueRange{});
    }

    auto copTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f32, "COp");
    auto aopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "AOp");
    auto bopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "BOp");
    IntegerAttr ldA = b.getIndexAttr(BK);   // A_sh leading dim
    IntegerAttr ldW = b.getIndexAttr(BK);   // W_sh leading dim
    IntegerAttr ldY = b.getIndexAttr(N);    // Y leading dim
    UnitAttr transpose = b.getUnitAttr();

    // nFrag fp32 accumulator fragments (one per the warp's column tiles), init 0.
    Value zeroF32 = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    SmallVector<Value> accInit;
    for (int j = 0; j < nFrag; ++j)
      accInit.push_back(
          b.create<gpu::SubgroupMmaConstantMatrixOp>(loc, copTy, zeroF32));

    // -- cp.async multistage pipeline ------------------------------------- //
    // Each K-tile of A[BM,BK] and W[BN,BK] is streamed global->shared by
    // nvgpu.device_async_copy (-> cp.async.cg, 16-byte/8-f16 vectors, L1
    // bypassed for the stream-once tiles), grouped + waited by
    // device_async_create_group / device_async_wait. N rotating shared buffers
    // let the loads of tiles K+1..K+N-1 overlap the WMMA compute of tile K.
    auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr dstElems = b.getIndexAttr(8);     // 8 f16 = 16 bytes per copy
    UnitAttr bypassL1 = b.getUnitAttr();          // cp.async.cg (skip L1)
    int64_t colsPerRow = BK / 8;                  // 4 vector-cols per 32-wide row
    // A tile is [BM,BK], W tile is [BN,BK]: when BM != BN the two tiles have a
    // different vector count, so each gets its own copies-per-thread (the old
    // shared r/cc was only valid for the BM==BN square tile).
    int64_t vPerThreadA = (BM * BK) / 8 / kBlock;
    int64_t vPerThreadW = (BN * BK) / 8 / kBlock;
    Value cCPR = b.create<arith::ConstantIndexOp>(loc, colsPerRow);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value cNStages = b.create<arith::ConstantIndexOp>(loc, nStages);
    Value cNm1 = b.create<arith::ConstantIndexOp>(loc, nStages - 1);

    // Issue the A+W async copies for K-offset `ktVal` into buffer `stageVal`;
    // return the group token. Vectorized 8-wide along K (contiguous in both A/W
    // global rows and the BK-wide shared rows -> 16-byte aligned cp.async).
    auto loadTile = [&](OpBuilder &nb, Location nl, Value ktVal,
                        Value stageVal) -> Value {
      SmallVector<Value> toks;
      // A[BM,BK] tile.
      for (int64_t it = 0; it < vPerThreadA; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * kBlock);
        Value vt = nb.create<arith::AddIOp>(nl, tx, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8);
        Value gk = nb.create<arith::AddIOp>(nl, ktVal, cc);
        Value ar = nb.create<arith::AddIOp>(nl, rowBlk, r);
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, aSh, ValueRange{stageVal, r, cc}, aMem,
            ValueRange{ar, gk}, dstElems, /*srcElements=*/Value(), bypassL1));
      }
      // W[BN,BK] tile.
      for (int64_t it = 0; it < vPerThreadW; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * kBlock);
        Value vt = nb.create<arith::AddIOp>(nl, tx, base);
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

    // Accumulate COp_j += A(stage) * B_j(stage) over the BK/16 k-subtiles of one
    // staged tile (rank-3 shared src indexed by the rotating `stageVal`).
    auto computeTile = [&](OpBuilder &nb, Location nl, Value stageVal,
                           ValueRange acc) -> SmallVector<Value> {
      SmallVector<Value> cur(acc.begin(), acc.end());
      for (int64_t ks = 0; ks < BK / kWMMA; ++ks) {
        Value ksc = nb.create<arith::ConstantIndexOp>(nl, ks * kWMMA);
        Value aFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
            nl, aopTy, aSh, ValueRange{stageVal, warpRow, ksc}, ldA,
            /*transpose=*/nullptr);
        for (int j = 0; j < nFrag; ++j) {
          Value cj = nb.create<arith::ConstantIndexOp>(nl, j * kWMMA);
          Value bFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
              nl, bopTy, wSh, ValueRange{stageVal, cj, ksc}, ldW, transpose);
          cur[j] = nb.create<gpu::SubgroupMmaComputeOp>(
              nl, copTy, aFrag, bFrag, cur[j], /*a_transpose=*/nullptr,
              /*b_transpose=*/nullptr);
        }
      }
      return cur;
    };

    // PROLOGUE: prefetch tiles 0..N-2 (none when N==1). Keep the last group's
    // token live -- it dominates the epilogue and is the (numGroups-driven) wait
    // dependency there (no token needs to be carried across the loop).
    Value prologTok;
    for (int s = 0; s < nStages - 1; ++s) {
      Value ktVal = b.create<arith::ConstantIndexOp>(loc, s * BK);
      Value stage = b.create<arith::ConstantIndexOp>(loc, s);
      prologTok = loadTile(b, loc, ktVal, stage);
    }

    // STEADY STATE: ki = 0..numK-N. Prefetch tile ki+N-1 into buffer
    // (ki+N-1)%N, wait for tile ki's group (keep N-1 groups in flight), compute
    // tile ki from buffer ki%N. The 4 accumulators ride the loop as iter_args.
    int64_t steadyTrips = numK - nStages + 1; // >= 1 (nStages <= numK)
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
          nb.create<gpu::BarrierOp>(nl); // async tile ki visible to all warps
          Value stageR = nb.create<arith::RemUIOp>(nl, ki, cNStages);
          SmallVector<Value> cur = computeTile(nb, nl, stageR, acc);
          nb.create<gpu::BarrierOp>(nl); // all reads done before buffer reuse
          nb.create<scf::YieldOp>(nl, cur);
        });

    // EPILOGUE: compute the last N-1 tiles (already prefetched), draining one
    // in-flight group per tile via a decreasing static numGroups. Static stage.
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

    if (!hasEpilogue) {
      // id: store each accumulator fragment directly to Y[myRow, colBlk+j*16]
      // (byte-identical to M7a -- no shared round-trip, no scalar epilogue).
      for (int j = 0; j < nFrag; ++j) {
        Value cj = b.create<arith::ConstantIndexOp>(loc, j * kWMMA);
        Value yCol = b.create<arith::AddIOp>(loc, colBlk, cj);
        b.create<gpu::SubgroupMmaStoreMatrixOp>(loc, finalAcc[j], yMem,
                                                ValueRange{myRow, yCol}, ldY,
                                                /*transpose=*/nullptr);
      }
    } else {
      // Fused epilogue: spill the 4 COp fragments to the [BM,BN] staging tile,
      // then the block cooperatively reloads every element, evaluates the
      // bias-add + activation scalar DAG, and writes Y.
      IntegerAttr ldC = b.getIndexAttr(BN);
      for (int j = 0; j < nFrag; ++j) {
        Value cj = b.create<arith::ConstantIndexOp>(loc, j * kWMMA);
        b.create<gpu::SubgroupMmaStoreMatrixOp>(loc, finalAcc[j], cSh,
                                                ValueRange{warpRow, cj}, ldC,
                                                /*transpose=*/nullptr);
      }
      b.create<gpu::BarrierOp>(loc);
      Value cBN = b.create<arith::ConstantIndexOp>(loc, BN);
      int64_t epIters = (BM * BN) / kBlock; // 4096/128 = 32 elems/thread
      for (int64_t it = 0; it < epIters; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value i = b.create<arith::AddIOp>(loc, tx, base);
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
            return failure(); // unsupported typed epilogue slipped past match()
          b.create<memref::StoreOp>(loc, out, outputMems[outputIndex],
                                    ValueRange{grow, gcol});
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

struct ConvertIntentContractionToGPUPass
    : ::intent::impl::ConvertIntentContractionToGPUBase<
          ConvertIntentContractionToGPUPass> {
  using ConvertIntentContractionToGPUBase::
      ConvertIntentContractionToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ContractionLowering probe(fn, (int)numStages, (int64_t)blockM,
                               (int64_t)blockN);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ContractionLowering low(fn, (int)numStages, (int64_t)blockM,
                              (int64_t)blockN);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("contraction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
