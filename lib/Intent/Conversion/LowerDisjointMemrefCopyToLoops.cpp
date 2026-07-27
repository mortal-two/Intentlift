//===- LowerDisjointMemrefCopyToLoops.cpp - standalone copy lowering -----===//

#include "Intent/Conversion/Common/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>

namespace intent {
#define GEN_PASS_DEF_LOWERDISJOINTMEMREFCOPYTOLOOPS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct LowerDisjointMemrefCopyToLoopsPass
    : ::intent::impl::LowerDisjointMemrefCopyToLoopsBase<
          LowerDisjointMemrefCopyToLoopsPass> {
  using LowerDisjointMemrefCopyToLoopsBase::
      LowerDisjointMemrefCopyToLoopsBase;

  void runOnOperation() override {
    llvm::SmallVector<memref::CopyOp> copies;
    getOperation().walk([&](memref::CopyOp copy) { copies.push_back(copy); });
    for (memref::CopyOp copy : copies) {
      auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
      auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
      // memref.copy defines overlapping source/destination locations as
      // undefined.  Consequently every valid ranked copy is already disjoint;
      // requiring the source SSA value to be defined directly by memref.alloc
      // was unnecessarily stricter than the operation contract and left ABI
      // argument copies as an external `memrefCopy` runtime dependency.
      if (!sourceType || !targetType ||
          sourceType.getShape() != targetType.getShape() ||
          sourceType.getElementType() != targetType.getElementType())
        continue;

      OpBuilder builder(copy);
      Location location = copy.getLoc();
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      llvm::SmallVector<Value> indices;
      std::function<void(unsigned)> emit = [&](unsigned dimension) {
        if (dimension == sourceType.getRank()) {
          Value value = builder.create<memref::LoadOp>(
              location, copy.getSource(), indices);
          builder.create<memref::StoreOp>(location, value, copy.getTarget(),
                                          indices);
          return;
        }
        Value upper;
        if (sourceType.isDynamicDim(dimension))
          upper = builder.create<memref::DimOp>(location, copy.getSource(),
                                                dimension);
        else
          upper = builder.create<arith::ConstantIndexOp>(
              location, sourceType.getDimSize(dimension));
        auto loop = builder.create<scf::ForOp>(location, zero, upper, one);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        indices.push_back(loop.getInductionVar());
        emit(dimension + 1);
        indices.pop_back();
      };
      emit(0);
      copy.erase();
    }
  }
};

} // namespace
