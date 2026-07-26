//===- ForwardBufferResultsToOutParams.cpp - remove result copy ---------===//
//
// A target-neutral cleanup for destination-passing CPU pipelines.  It runs
// after upstream buffer-results-to-out-params and forwards an exact local
// result buffer into the corresponding entry block output argument.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_FORWARDBUFFERRESULTSTOOUTPARAMS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct ForwardBufferResultsToOutParamsPass
    : ::intent::impl::ForwardBufferResultsToOutParamsBase<
          ForwardBufferResultsToOutParamsPass> {
  using ForwardBufferResultsToOutParamsBase::
      ForwardBufferResultsToOutParamsBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp function) {
      if (function.isDeclaration() || function.empty())
        return;
      Block &entry = function.front();
      llvm::SmallVector<memref::AllocOp> allocations;
      function.walk(
          [&](memref::AllocOp allocation) { allocations.push_back(allocation); });

      for (memref::AllocOp allocation : allocations) {
        memref::CopyOp resultCopy;
        bool invalid = false;
        for (Operation *user : allocation.getResult().getUsers()) {
          auto copy = dyn_cast<memref::CopyOp>(user);
          if (!copy || copy.getSource() != allocation.getResult())
            continue;
          if (resultCopy) {
            invalid = true;
            break;
          }
          resultCopy = copy;
        }
        if (invalid || !resultCopy)
          continue;
        auto destination = dyn_cast<BlockArgument>(resultCopy.getTarget());
        if (!destination || destination.getOwner() != &entry ||
            destination.getType() != allocation.getType())
          continue;

        allocation.getResult().replaceUsesWithIf(
            destination, [&](OpOperand &use) {
              return use.getOwner() != resultCopy.getOperation();
            });
        resultCopy.erase();
        if (allocation->use_empty())
          allocation.erase();
      }
    });
  }
};

} // namespace
