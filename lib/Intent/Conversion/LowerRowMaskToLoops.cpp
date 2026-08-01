//===- LowerRowMaskToLoops.cpp - hoisted typed row-mask loops ------------===//
//
// This pass consumes the target-neutral bufferized shape of a row broadcast.
// It never names a source kernel and emits only standard MLIR loops/memrefs.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent {
#define GEN_PASS_DEF_LOWERROWMASKTOLOOPS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct RowMaskMatch {
  linalg::GenericOp cast;
  linalg::GenericOp product;
  memref::AllocOp castBuffer;
  Value data;
  Value mask;
  Value output;
  MemRefType dataType;
};

static bool isIdentity(AffineMap map, unsigned rank) {
  return map.getNumDims() == rank && map.getNumSymbols() == 0 &&
         map.getNumResults() == rank && map.isIdentity();
}

static bool isRowProjection(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 &&
         map.getResult(0) == getAffineDimExpr(0, map.getContext());
}

static bool hasOnlyUsers(Value value, ArrayRef<Operation *> expected) {
  llvm::SmallPtrSet<Operation *, 4> expectedSet(expected.begin(), expected.end());
  llvm::SmallPtrSet<Operation *, 4> actual;
  for (Operation *user : value.getUsers())
    actual.insert(user);
  return actual == expectedSet;
}

static bool hasParallelIterators(linalg::GenericOp op, unsigned count) {
  ArrayRef<utils::IteratorType> iterators = op.getIteratorTypesArray();
  return iterators.size() == count &&
         llvm::all_of(iterators, [](utils::IteratorType iterator) {
           return iterator == utils::IteratorType::parallel;
         });
}

static bool hasIdentityLayout(MemRefType type) {
  return type.getLayout().isIdentity();
}

static std::optional<RowMaskMatch> matchRowMask(linalg::GenericOp product) {
  if (!product || !product.hasPureBufferSemantics() ||
      product.getNumLoops() != 2 || product.getNumDpsInputs() != 2 ||
      product.getNumDpsInits() != 1 || product->getNumResults() != 0 ||
      !hasParallelIterators(product, 2))
    return std::nullopt;

  SmallVector<AffineMap> maps = product.getIndexingMapsArray();
  if (maps.size() != 3 || !isIdentity(maps[2], 2))
    return std::nullopt;
  unsigned dataIndex;
  unsigned maskIndex;
  if (isIdentity(maps[0], 2) && isRowProjection(maps[1])) {
    dataIndex = 0;
    maskIndex = 1;
  } else if (isRowProjection(maps[0]) && isIdentity(maps[1], 2)) {
    dataIndex = 1;
    maskIndex = 0;
  } else {
    return std::nullopt;
  }

  Block &productBody = product.getRegion().front();
  if (productBody.getNumArguments() != 3 ||
      !productBody.getArgument(2).use_empty() ||
      !llvm::hasSingleElement(productBody.without_terminator()))
    return std::nullopt;
  auto multiply = dyn_cast<arith::MulFOp>(&productBody.front());
  auto productYield = dyn_cast<linalg::YieldOp>(productBody.getTerminator());
  if (!multiply || !productYield || productYield.getValues().size() != 1 ||
      productYield.getValues().front() != multiply.getResult())
    return std::nullopt;
  Value dataScalar = productBody.getArgument(dataIndex);
  Value maskScalar = productBody.getArgument(maskIndex);
  if (!((multiply.getLhs() == dataScalar && multiply.getRhs() == maskScalar) ||
        (multiply.getLhs() == maskScalar && multiply.getRhs() == dataScalar)))
    return std::nullopt;

  Value data = product.getDpsInputs()[dataIndex];
  Value castValue = product.getDpsInputs()[maskIndex];
  Value output = product.getDpsInits()[0];
  auto dataType = dyn_cast<MemRefType>(data.getType());
  auto castType = dyn_cast<MemRefType>(castValue.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!dataType || !castType || !outputType || dataType.getRank() != 2 ||
      castType.getRank() != 1 || outputType != dataType ||
      !dataType.getElementType().isF32() ||
      !castType.getElementType().isF32() || !dataType.hasStaticShape() ||
      castType.getDimSize(0) != dataType.getDimSize(0) ||
      !hasIdentityLayout(dataType) || !hasIdentityLayout(castType))
    return std::nullopt;

  auto castBuffer = castValue.getDefiningOp<memref::AllocOp>();
  if (!castBuffer)
    return std::nullopt;
  linalg::GenericOp cast;
  for (Operation *user : castValue.getUsers())
    if (auto generic = dyn_cast<linalg::GenericOp>(user))
      if (generic != product)
        cast = generic;
  if (!cast || !cast.hasPureBufferSemantics() || cast.getNumLoops() != 1 ||
      cast.getNumDpsInputs() != 1 || cast.getNumDpsInits() != 1 ||
      cast->getNumResults() != 0 || cast.getDpsInits()[0] != castValue ||
      !hasParallelIterators(cast, 1) ||
      !hasOnlyUsers(castValue, {cast.getOperation(), product.getOperation()}))
    return std::nullopt;
  SmallVector<AffineMap> castMaps = cast.getIndexingMapsArray();
  if (castMaps.size() != 2 || !isIdentity(castMaps[0], 1) ||
      !isIdentity(castMaps[1], 1))
    return std::nullopt;
  Value mask = cast.getDpsInputs()[0];
  auto maskType = dyn_cast<MemRefType>(mask.getType());
  if (!maskType || maskType.getRank() != 1 ||
      !maskType.getElementType().isInteger(1) ||
      maskType.getShape() != castType.getShape() ||
      !hasIdentityLayout(maskType))
    return std::nullopt;
  Block &castBody = cast.getRegion().front();
  if (castBody.getNumArguments() != 2 ||
      !castBody.getArgument(1).use_empty() ||
      !llvm::hasSingleElement(castBody.without_terminator()))
    return std::nullopt;
  auto conversion = dyn_cast<arith::UIToFPOp>(&castBody.front());
  auto castYield = dyn_cast<linalg::YieldOp>(castBody.getTerminator());
  if (!conversion || conversion.getIn() != castBody.getArgument(0) ||
      !castYield || castYield.getValues().size() != 1 ||
      castYield.getValues().front() != conversion.getResult())
    return std::nullopt;

  return RowMaskMatch{cast, product, castBuffer, data, mask, output, dataType};
}

static void emitInnerLoop(OpBuilder &builder, Location location,
                          const RowMaskMatch &match, Value row,
                          bool copyInput, unsigned vectorWidth,
                          Value factor = {}) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  int64_t width = match.dataType.getDimSize(1);
  Value iterations = builder.create<arith::ConstantIndexOp>(
      location, width / static_cast<int64_t>(vectorWidth));
  auto loop = builder.create<scf::ForOp>(location, zero, iterations, one);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
  Value column = loop.getInductionVar();
  if (vectorWidth > 1) {
    Value packetWidth = bodyBuilder.create<arith::ConstantIndexOp>(
        location, vectorWidth);
    column = bodyBuilder.create<arith::MulIOp>(location, column, packetWidth);
  }
  Value result;
  if (copyInput || factor) {
    Value input =
        vectorWidth == 1
            ? bodyBuilder.create<memref::LoadOp>(
                  location, match.data, ValueRange{row, column})
                  .getResult()
            : bodyBuilder.create<vector::LoadOp>(
                  location,
                  VectorType::get({static_cast<int64_t>(vectorWidth)},
                                  bodyBuilder.getF32Type()),
                  match.data, ValueRange{row, column})
                  .getResult();
    if (factor && vectorWidth > 1)
      factor = bodyBuilder.create<vector::SplatOp>(location, input.getType(), factor);
    result = factor ? bodyBuilder.create<arith::MulFOp>(location, input, factor)
                    : input;
  } else {
    Value scalarZero = bodyBuilder.create<arith::ConstantOp>(
        location, bodyBuilder.getF32FloatAttr(0.0));
    result = vectorWidth == 1
                 ? scalarZero
                 : bodyBuilder.create<vector::SplatOp>(
                       location,
                       VectorType::get({static_cast<int64_t>(vectorWidth)},
                                       bodyBuilder.getF32Type()),
                       scalarZero)
                       .getResult();
  }
  if (vectorWidth == 1)
    bodyBuilder.create<memref::StoreOp>(location, result, match.output,
                                        ValueRange{row, column});
  else
    bodyBuilder.create<vector::StoreOp>(location, result, match.output,
                                        ValueRange{row, column});
}

static void lowerRowMask(RowMaskMatch match, bool zeroFill,
                         unsigned vectorWidth) {
  Location location = match.product.getLoc();
  OpBuilder builder(match.product);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rows = builder.create<arith::ConstantIndexOp>(
      location, match.dataType.getDimSize(0));
  auto outer = builder.create<scf::ForOp>(location, zero, rows, one);
  OpBuilder rowBuilder = OpBuilder::atBlockBegin(outer.getBody());
  Value row = outer.getInductionVar();
  Value mask = rowBuilder.create<memref::LoadOp>(
      location, match.mask, ValueRange{row});
  if (zeroFill) {
    auto branch = rowBuilder.create<scf::IfOp>(location, mask,
                                               /*withElseRegion=*/true);
    OpBuilder thenBuilder = branch.getThenBodyBuilder();
    emitInnerLoop(thenBuilder, location, match, row, /*copyInput=*/true,
                  vectorWidth);
    OpBuilder elseBuilder = branch.getElseBodyBuilder();
    emitInnerLoop(elseBuilder, location, match, row, /*copyInput=*/false,
                  vectorWidth);
  } else {
    Value factor = rowBuilder.create<arith::UIToFPOp>(
        location, rowBuilder.getF32Type(), mask);
    emitInnerLoop(rowBuilder, location, match, row, /*copyInput=*/true,
                  vectorWidth, factor);
  }

  match.product.erase();
  match.cast.erase();
  match.castBuffer.erase();
}

struct LowerRowMaskToLoopsPass
    : ::intent::impl::LowerRowMaskToLoopsBase<LowerRowMaskToLoopsPass> {
  using LowerRowMaskToLoopsBase::LowerRowMaskToLoopsBase;

  void runOnOperation() override {
    if (vectorWidth != 1 && vectorWidth != 2 && vectorWidth != 4 &&
        vectorWidth != 8) {
      getOperation().emitError("row-mask vector width must be 1, 2, 4, or 8");
      signalPassFailure();
      return;
    }
    SmallVector<RowMaskMatch> matches;
    getOperation().walk([&](linalg::GenericOp generic) {
      if (std::optional<RowMaskMatch> match = matchRowMask(generic))
        matches.push_back(*match);
    });
    if (matches.empty()) {
      if (requireLowering) {
        getOperation().emitError(
            "required typed row-mask graph was not present after bufferization");
        signalPassFailure();
      }
      return;
    }
    for (RowMaskMatch match : matches) {
      if (match.dataType.getDimSize(1) % vectorWidth != 0) {
        match.product.emitError(
            "row width is not divisible by the requested vector width");
        signalPassFailure();
        return;
      }
      lowerRowMask(match, zeroFill, vectorWidth);
    }
    Builder builder(getOperation().getContext());
    getOperation()->setAttr("intent.rvv.rowmask_lowering_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.rowmask_zero_fill_count",
                            builder.getI64IntegerAttr(zeroFill ? matches.size() : 0));
    getOperation()->setAttr("intent.rvv.rowmask_mask_hoist_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.rowmask_vector_width",
                            builder.getI64IntegerAttr(vectorWidth));
  }
};

} // namespace
