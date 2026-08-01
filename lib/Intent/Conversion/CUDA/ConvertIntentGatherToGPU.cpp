//===- ConvertIntentGatherToGPU.cpp - gather/scatter -> gpu.launch --------===//
//
// The C3 data-dependent-addressing pass. Lowers a func whose body is exactly ONE
// `intent.gather` (torch.gather / index_select / embedding row-gather) or ONE
// `intent.scatter` (torch.scatter / scatter_add_) over rank-2 data to a single
// block-cooperative `gpu.launch`: one block per output/scatter row, the column
// axis split over the block's T threads (K = cols/T columns each, unrolled).
//
// The NEW mechanism vs. the elementwise/reduce passes: the per-element access
// coordinate on `data` is a RUNTIME value read from the integer `index` memref —
// `idx = index_cast(memref.load index[...])` — used directly as a
// `memref.load`/`memref.store`/`memref.atomic_rmw` coordinate on `data`. This is
// a data-dependent gather/scatter address, not an affine function of the loop.
//
//   gather : destination-passing. new signature = (data, index) ++ [Ro,Co] out.
//            out[bx,col] = data[<coord with the index axis read from index>].
//   scatter: IN-PLACE on the (caller-cloned) data memref. new signature =
//            (data, index, src). data[<coord with axis=index[bx,col]>] (op)=
//            src[bx,col]; op = store (reduce none) or atomic_rmw addf (add).
//
// It additionally recognizes a typed two-level chain
//   take(data,row_index,axis=0) -> along_axis(tmp,col_index,axis=1).
// The canonical state eagerly evaluates the complete first-gather row into
// workgroup memory before selecting its final element.  The Full-only composed
// state composes both runtime index maps and directly loads data[row,col].
//
// Correctness-only (the index stream is exact -> bit-faithful). Static shapes,
// cols % T == 0 (tail-mask is a follow-up). The two match() predicates for the
// existing passes reject any body with a gather/scatter op (they only accept
// map/reduce/select/iota), so this pass is disjoint and composes freely.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTGATHERTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's gather/scatter lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct GatherScatterLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  bool requestComposeGatherChain;

  // populated by match()
  GatherOp gather;    // set for a gather func
  ScatterOp scatter;  // set for a scatter func
  GatherOp firstGather;
  GatherOp secondGather;
  tensor::ExpandShapeOp expandedColumnIndex;
  tensor::CollapseShapeOp collapsedResult;
  bool gatherChain = false;
  int64_t axis = 0;
  // Iteration geometry: one block per row (Ro), T threads over Co columns, K
  // columns per thread. For gather these are the OUTPUT dims; for scatter the
  // SRC/index dims (that is what drives the writes).
  int64_t Ro = 0, Co = 0, T = 0, K = 0;

  GatherScatterLowering(func::FuncOp fn, unsigned bt, bool composeChain)
      : fn(fn), blockThreads(bt),
        requestComposeGatherChain(composeChain) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult setGeometry(int64_t rows, int64_t cols) {
    Ro = rows;
    Co = cols;
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Co);
    if (T <= 0 || Co % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented follow-up
    K = Co / T;
    return success();
  }

  LogicalResult matchGatherChain(Block &body, func::ReturnOp ret) {
    if (ret.getNumOperands() != 1)
      return failure();
    SmallVector<Operation *> operations;
    for (Operation &op : body.without_terminator())
      operations.push_back(&op);
    if (operations.size() != 4)
      return failure();

    auto first = dyn_cast<GatherOp>(operations[0]);
    auto expand = dyn_cast<tensor::ExpandShapeOp>(operations[1]);
    auto second = dyn_cast<GatherOp>(operations[2]);
    auto collapse = dyn_cast<tensor::CollapseShapeOp>(operations[3]);
    if (!first || !expand || !second || !collapse ||
        first.getMode() != "take" || first.getAxis() != 0 ||
        second.getMode() != "along_axis" || second.getAxis() != 1 ||
        second.getData() != first.getResult() ||
        second.getIndex() != expand.getResult() ||
        collapse.getSrc() != second.getResult() ||
        ret.getOperand(0) != collapse.getResult())
      return failure();

    auto dataTy = dyn_cast<RankedTensorType>(first.getData().getType());
    auto rowIndexTy = dyn_cast<RankedTensorType>(first.getIndex().getType());
    auto firstResultTy = dyn_cast<RankedTensorType>(first.getResult().getType());
    auto colIndexTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    auto expandedIndexTy = dyn_cast<RankedTensorType>(expand.getResult().getType());
    auto secondResultTy = dyn_cast<RankedTensorType>(second.getResult().getType());
    auto resultTy = dyn_cast<RankedTensorType>(collapse.getResult().getType());
    if (!dataTy || !rowIndexTy || !firstResultTy || !colIndexTy ||
        !expandedIndexTy || !secondResultTy || !resultTy ||
        !dataTy.hasStaticShape() || !rowIndexTy.hasStaticShape() ||
        !firstResultTy.hasStaticShape() || !colIndexTy.hasStaticShape() ||
        !expandedIndexTy.hasStaticShape() || !secondResultTy.hasStaticShape() ||
        !resultTy.hasStaticShape() || dataTy.getRank() != 2 ||
        rowIndexTy.getRank() != 1 || colIndexTy.getRank() != 1 ||
        firstResultTy.getRank() != 2 || expandedIndexTy.getRank() != 2 ||
        secondResultTy.getRank() != 2 || resultTy.getRank() != 1)
      return failure();

    int64_t rows = rowIndexTy.getDimSize(0);
    int64_t columns = dataTy.getDimSize(1);
    if (rows <= 0 || columns <= 0 ||
        firstResultTy.getShape() != ArrayRef<int64_t>({rows, columns}) ||
        colIndexTy.getDimSize(0) != rows ||
        expandedIndexTy.getShape() != ArrayRef<int64_t>({rows, 1}) ||
        secondResultTy.getShape() != ArrayRef<int64_t>({rows, 1}) ||
        resultTy.getDimSize(0) != rows ||
        firstResultTy.getElementType() != dataTy.getElementType() ||
        secondResultTy.getElementType() != dataTy.getElementType() ||
        resultTy.getElementType() != dataTy.getElementType() ||
        !isa<IntegerType>(rowIndexTy.getElementType()) ||
        !isa<IntegerType>(colIndexTy.getElementType()))
      return failure();

    Ro = rows;
    Co = columns;
    T = blockThreads ? static_cast<int64_t>(blockThreads) : autoThreads(columns);
    if (T < 32 || T > 1024 || T % 32 != 0)
      return failure();
    K = (Co + T - 1) / T;
    firstGather = first;
    secondGather = second;
    expandedColumnIndex = expand;
    collapsedResult = collapse;
    gatherChain = true;
    return success();
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    if (succeeded(matchGatherChain(body, ret)))
      return success();
    if (requestComposeGatherChain)
      return failure();
    // Exactly one non-terminator op, and it produces the returned value.
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    if (!only || ret.getOperand(0) != only->getResult(0))
      return failure();

    if (auto g = dyn_cast<GatherOp>(only)) {
      auto dataTy = dyn_cast<RankedTensorType>(g.getData().getType());
      auto indexTy = dyn_cast<RankedTensorType>(g.getIndex().getType());
      auto resultTy = dyn_cast<RankedTensorType>(g.getResult().getType());
      int64_t candidateAxis = g.getAxis();
      StringRef mode = g.getMode();
      if (!dataTy || !indexTy || !resultTy ||
          !dataTy.hasStaticShape() || !indexTy.hasStaticShape() ||
          !resultTy.hasStaticShape() || dataTy.getRank() != 2 ||
          resultTy.getRank() != 2 ||
          (candidateAxis != 0 && candidateAxis != 1) ||
          (mode != "take" && mode != "along_axis") ||
          !isa<IntegerType>(indexTy.getElementType()) ||
          resultTy.getElementType() != dataTy.getElementType())
        return failure();

      if (mode == "take") {
        if (indexTy.getRank() != 1)
          return failure();
        SmallVector<int64_t> expectedShape(dataTy.getShape());
        expectedShape[candidateAxis] = indexTy.getDimSize(0);
        if (resultTy.getShape() != ArrayRef<int64_t>(expectedShape))
          return failure();
      } else {
        if (indexTy.getRank() != 2 ||
            resultTy.getShape() != indexTy.getShape())
          return failure();
        int64_t preservedAxis = candidateAxis == 0 ? 1 : 0;
        if (resultTy.getDimSize(preservedAxis) >
            dataTy.getDimSize(preservedAxis))
          return failure();
      }
      if (dataTy.getDimSize(0) <= 0 || dataTy.getDimSize(1) <= 0 ||
          resultTy.getDimSize(0) <= 0 || resultTy.getDimSize(1) <= 0)
        return failure();
      gather = g;
      axis = candidateAxis;
      return setGeometry(resultTy.getDimSize(0), resultTy.getDimSize(1));
    }
    if (auto s = dyn_cast<ScatterOp>(only)) {
      auto dataTy = dyn_cast<RankedTensorType>(s.getData().getType());
      auto indexTy = dyn_cast<RankedTensorType>(s.getIndex().getType());
      auto srcTy = dyn_cast<RankedTensorType>(s.getSrc().getType());
      auto resultTy = dyn_cast<RankedTensorType>(s.getResult().getType());
      int64_t candidateAxis = s.getAxis();
      if (!dataTy || !indexTy || !srcTy || !resultTy ||
          !dataTy.hasStaticShape() || !indexTy.hasStaticShape() ||
          !srcTy.hasStaticShape() || !resultTy.hasStaticShape() ||
          dataTy.getRank() != 2 || indexTy.getRank() != 2 ||
          srcTy.getRank() != 2 || resultTy.getRank() != 2 ||
          (candidateAxis != 0 && candidateAxis != 1) ||
          s.getMode() != "along_axis" ||
          (s.getReduce() != "none" && s.getReduce() != "add") ||
          (s.getReduce() == "add" &&
           !isa<FloatType>(dataTy.getElementType())) ||
          !isa<IntegerType>(indexTy.getElementType()) ||
          srcTy.getShape() != indexTy.getShape() ||
          resultTy.getShape() != dataTy.getShape() ||
          srcTy.getElementType() != dataTy.getElementType() ||
          resultTy.getElementType() != dataTy.getElementType())
        return failure();
      int64_t preservedAxis = candidateAxis == 0 ? 1 : 0;
      if (dataTy.getDimSize(0) <= 0 || dataTy.getDimSize(1) <= 0 ||
          srcTy.getDimSize(preservedAxis) >
              dataTy.getDimSize(preservedAxis) ||
          srcTy.getDimSize(0) <= 0 || srcTy.getDimSize(1) <= 0)
        return failure();
      scatter = s;
      axis = candidateAxis;
      return setGeometry(srcTy.getDimSize(0), srcTy.getDimSize(1));
    }
    return failure();
  }

  // ---- emission ---------------------------------------------------------- //
  // Load index[coord] and cast it to an `index`-typed value for use as a
  // data-memref coordinate. `coord` is 1-D (take: index is rank-1) or 2-D.
  Value loadIndex(OpBuilder &b, Location loc, Value idxMem, ValueRange coord) {
    Value raw = b.create<memref::LoadOp>(loc, idxMem, coord);
    if (raw.getType().isIndex())
      return raw;
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), raw);
  }

  void attachGatherChainState(func::FuncOp lowered, StringRef state) {
    MLIRContext *ctx = lowered.getContext();
    auto i64 = IntegerType::get(ctx, 64);
    bool composed = state.contains("composed");
    lowered->setAttr("intent.backend.lowering_path_kind",
                     StringAttr::get(ctx, composed ? "gather_composed"
                                                   : "gather"));
    lowered->setAttr("intent.backend.reduction_topology",
                     StringAttr::get(ctx, "none"));
    lowered->setAttr("intent.backend.approximation_mode",
                     StringAttr::get(ctx, "none"));
    lowered->setAttr(
        "intent.backend.transformation_trace",
        ArrayAttr::get(ctx, {StringAttr::get(ctx, state)}));
    lowered->setAttr("intent.backend.block_threads",
                     IntegerAttr::get(i64, T));
    lowered->setAttr("intent.backend.grid_blocks",
                     IntegerAttr::get(i64, composed ? (Ro + T - 1) / T : Ro));
    lowered->setAttr("intent.backend.vector_width", IntegerAttr::get(i64, 1));
    for (StringRef name : {"fma_count", "shuffle_count", "all_reduce_count",
                           "predicated_value_if_count",
                           "folded_constexpr_select_count"})
      lowered->setAttr(("intent.backend." + name).str(),
                       IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.barrier_count",
                     IntegerAttr::get(i64, composed ? 0 : 1));
    lowered->setAttr("intent.backend.gather_chain_depth",
                     IntegerAttr::get(i64, 2));
  }

  LogicalResult rewriteGatherChainMaterialized() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto dataTy = cast<RankedTensorType>(firstGather.getData().getType());
    auto rowIndexTy = cast<RankedTensorType>(firstGather.getIndex().getType());
    auto colIndexTy = cast<RankedTensorType>(expandedColumnIndex.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(collapsedResult.getResult().getType());
    Type elementType = dataTy.getElementType();
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0)
      return failure();

    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(rowIndexTy),
                            memrefOf(colIndexTy), memrefOf(resultTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value rowIndexMem = entry->getArgument(1);
    Value colIndexMem = entry->getArgument(2);
    Value outMem = entry->getArgument(3);

    Value cRows = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cThreads = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sharedBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr(static_cast<int32_t>(Co * bitWidth / 8)));
    auto launch = b.create<gpu::LaunchOp>(loc, cRows, c1, c1, cThreads,
                                         c1, c1, sharedBytes);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cColumns = b.create<arith::ConstantIndexOp>(loc, Co);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto rowTy = MemRefType::get({Co}, elementType, nullptr, wgSpace);
    Value rowScratch =
        b.create<memref::ViewOp>(loc, rowTy, raw, c0, ValueRange{});

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value offset = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, offset);
      }
      Value active = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, col, cColumns);
      auto guarded = b.create<scf::IfOp>(loc, active, false);
      OpBuilder thenBuilder = guarded.getThenBodyBuilder();
      Value row = loadIndex(thenBuilder, loc, rowIndexMem, ValueRange{bx});
      Value value = thenBuilder.create<memref::LoadOp>(
          loc, dataMem, ValueRange{row, col});
      thenBuilder.create<memref::StoreOp>(loc, value, rowScratch,
                                          ValueRange{col});
    }
    b.create<gpu::BarrierOp>(loc);
    Value leader = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, tx, c0);
    auto leaderOnly = b.create<scf::IfOp>(loc, leader, false);
    OpBuilder leaderBuilder = leaderOnly.getThenBodyBuilder();
    Value selectedColumn =
        loadIndex(leaderBuilder, loc, colIndexMem, ValueRange{bx});
    Value value = leaderBuilder.create<memref::LoadOp>(
        loc, rowScratch, ValueRange{selectedColumn});
    leaderBuilder.create<memref::StoreOp>(loc, value, outMem, ValueRange{bx});

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachGatherChainState(gpuFn, "cuda.gather-chain.materialized-row.v1");
    fn.erase();
    return success();
  }

  LogicalResult rewriteGatherChainComposed() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto dataTy = cast<RankedTensorType>(firstGather.getData().getType());
    auto rowIndexTy = cast<RankedTensorType>(firstGather.getIndex().getType());
    auto colIndexTy = cast<RankedTensorType>(expandedColumnIndex.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(collapsedResult.getResult().getType());
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(rowIndexTy),
                            memrefOf(colIndexTy), memrefOf(resultTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value rowIndexMem = entry->getArgument(1);
    Value colIndexMem = entry->getArgument(2);
    Value outMem = entry->getArgument(3);

    int64_t blocks = (Ro + T - 1) / T;
    Value cBlocks = b.create<arith::ConstantIndexOp>(loc, blocks);
    Value cThreads = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch =
        b.create<gpu::LaunchOp>(loc, cBlocks, c1, c1, cThreads, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value linear = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bx, cT), tx);
    Value cRows = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value active = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, linear, cRows);
    auto guarded = b.create<scf::IfOp>(loc, active, false);
    OpBuilder thenBuilder = guarded.getThenBodyBuilder();
    Value row =
        loadIndex(thenBuilder, loc, rowIndexMem, ValueRange{linear});
    Value column =
        loadIndex(thenBuilder, loc, colIndexMem, ValueRange{linear});
    Value value = thenBuilder.create<memref::LoadOp>(
        loc, dataMem, ValueRange{row, column});
    thenBuilder.create<memref::StoreOp>(loc, value, outMem,
                                        ValueRange{linear});

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachGatherChainState(gpuFn, "cuda.gather-chain.composed-index.v1");
    fn.erase();
    return success();
  }

  LogicalResult rewriteGather() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(gather.getData().getType());
    auto idxTy = cast<RankedTensorType>(gather.getIndex().getType());
    auto outTy = cast<RankedTensorType>(gather.getResult().getType());
    bool alongAxis = gather.getMode() == "along_axis";

    // New signature: data, index (as memrefs) ++ one [Ro,Co] output memref.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(outTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value outMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      // Data-dependent load coordinate: read the index, replace the `axis`
      // coordinate of the data access with it.
      Value idx;
      if (alongAxis) // index is [Ro,Co]: read at the output coordinate
        idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      else           // take: index is rank-1, selecting along `axis`
        idx = loadIndex(b, loc, idxMem, ValueRange{axis == 0 ? bx : col});
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      Value v = b.create<memref::LoadOp>(loc, dataMem, ValueRange{r0, r1});
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewriteScatter() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(scatter.getData().getType());
    auto idxTy = cast<RankedTensorType>(scatter.getIndex().getType());
    auto srcTy = cast<RankedTensorType>(scatter.getSrc().getType());
    bool add = scatter.getReduce() == "add";

    // New signature: data (in/out, caller-cloned), index, src — all memrefs.
    // Scatter is IN-PLACE on `data`: no appended output arg.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(srcTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value srcMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      Value v = b.create<memref::LoadOp>(loc, srcMem, ValueRange{bx, col});
      Value idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      // Data-dependent store coordinate on `data`: replace the `axis` coord.
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      if (add)
        b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v,
                                      dataMem, ValueRange{r0, r1});
      else
        b.create<memref::StoreOp>(loc, v, dataMem, ValueRange{r0, r1});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (gatherChain)
      return requestComposeGatherChain ? rewriteGatherChainComposed()
                                       : rewriteGatherChainMaterialized();
    if (gather)
      return rewriteGather();
    return rewriteScatter();
  }
};

struct ConvertIntentGatherToGPUPass
    : ::intent::impl::ConvertIntentGatherToGPUBase<
          ConvertIntentGatherToGPUPass> {
  using ConvertIntentGatherToGPUBase::ConvertIntentGatherToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      GatherScatterLowering probe(fn, blockThreads, composeGatherChain);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      GatherScatterLowering low(fn, blockThreads, composeGatherChain);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("gather/scatter lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
