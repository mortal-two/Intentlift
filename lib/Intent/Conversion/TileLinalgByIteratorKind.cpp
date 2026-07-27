//===- TileLinalgByIteratorKind.cpp - generic iterator-aware tiling -------===//
//
// A target-neutral, source-blind schedule transformation.  It consumes only
// concrete compiler schedule parameters and the iterator kinds already present
// in Linalg IR.  C may order those parameters outside this pass; it never
// changes matching or legality here.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace intent {
#define GEN_PASS_DEF_TILELINALGBYITERATORKIND
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct TileLinalgByIteratorKindPass
    : ::intent::impl::TileLinalgByIteratorKindBase<
          TileLinalgByIteratorKindPass> {
  using TileLinalgByIteratorKindBase::TileLinalgByIteratorKindBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (parallelTile == 0 && reductionTile == 0) {
      module->setAttr("intent.schedule.linalg_tiled_op_count",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));
      return;
    }

    SmallVector<Operation *> worklist;
    module.walk([&](linalg::LinalgOp op) {
      if (op.hasPureTensorSemantics())
        worklist.push_back(op.getOperation());
    });

    IRRewriter rewriter(&getContext());
    unsigned tiledOpCount = 0;
    unsigned generatedLoopCount = 0;
    bool parallelTileApplied = false;
    bool reductionTileApplied = false;
    for (Operation *operation : worklist) {
      if (!operation->getBlock())
        continue;
      auto linalgOp = dyn_cast<linalg::LinalgOp>(operation);
      auto tilingOp = dyn_cast<TilingInterface>(operation);
      if (!linalgOp || !tilingOp)
        continue;

      SmallVector<OpFoldResult> tileSizes;
      bool hasNonZeroTile = false;
      bool opUsesParallelTile = false;
      bool opUsesReductionTile = false;
      for (utils::IteratorType iterator : linalgOp.getIteratorTypesArray()) {
        unsigned tile = iterator == utils::IteratorType::reduction
                            ? reductionTile
                            : parallelTile;
        tileSizes.push_back(rewriter.getIndexAttr(tile));
        hasNonZeroTile |= tile != 0;
        opUsesReductionTile |=
            tile != 0 && iterator == utils::IteratorType::reduction;
        opUsesParallelTile |=
            tile != 0 && iterator != utils::IteratorType::reduction;
      }
      if (!hasNonZeroTile || tileSizes.empty())
        continue;

      rewriter.setInsertionPoint(operation);
      scf::SCFTilingOptions options;
      options.setTileSizes(tileSizes).setLoopType(
          scf::SCFTilingOptions::LoopType::ForOp);
      FailureOr<scf::SCFTilingResult> tiled =
          scf::tileUsingSCF(rewriter, tilingOp, options);
      if (failed(tiled)) {
        operation->emitError("failed iterator-kind Linalg tiling");
        signalPassFailure();
        return;
      }
      if (tiled->mergeResult.replacements.size() != operation->getNumResults()) {
        operation->emitError("tiling produced an incomplete replacement set");
        signalPassFailure();
        return;
      }
      generatedLoopCount += tiled->loops.size();
      ++tiledOpCount;
      parallelTileApplied |= opUsesParallelTile;
      reductionTileApplied |= opUsesReductionTile;
      rewriter.replaceOp(operation, tiled->mergeResult.replacements);
    }

    Builder builder(&getContext());
    module->setAttr("intent.schedule.parallel_tile",
                    builder.getI64IntegerAttr(parallelTileApplied ? parallelTile
                                                                 : 0));
    module->setAttr("intent.schedule.reduction_tile",
                    builder.getI64IntegerAttr(
                        reductionTileApplied ? reductionTile : 0));
    module->setAttr("intent.schedule.linalg_tiled_op_count",
                    builder.getI64IntegerAttr(tiledOpCount));
    module->setAttr("intent.schedule.generated_loop_count",
                    builder.getI64IntegerAttr(generatedLoopCount));
  }
};

} // namespace
