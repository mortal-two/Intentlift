//===- LowerPairedRotationToLoops.cpp - paired rotation to direct loops ---===//
//
// Consumes the closed, target-neutral paired-rotation view attached by the J
// materializer.  The pass runs after bufferization and replaces the exploded
// gather/cast/multiply/scatter graph with direct SCF/memref code.  It does not
// inspect a kernel name or emit target source; LLVM remains responsible for
// selecting RVV instructions.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent {
#define GEN_PASS_DEF_LOWERPAIREDROTATIONTOLOOPS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct InterleavedRotationMatch {
  func::FuncOp function;
  Value query;
  Value key;
  Value angles;
  Value imaginarySign;
  Value queryOutput;
  Value keyOutput;
  MemRefType queryType;
  MemRefType keyType;
  MemRefType angleType;
  int64_t batch;
  int64_t sequence;
  int64_t queryHeads;
  int64_t keyHeads;
  int64_t headDimension;
};

static bool hasIdentityStaticLayout(MemRefType type) {
  return type && type.hasStaticShape() && type.getLayout().isIdentity();
}

static std::optional<InterleavedRotationMatch>
matchInterleavedRotation(func::FuncOp function) {
  auto version =
      function->getAttrOfType<StringAttr>("intent.paired_rotation_version");
  auto layout = function->getAttrOfType<StringAttr>("intent.paired_rotation");
  auto inputs = function->getAttrOfType<DenseI64ArrayAttr>(
      "intent.paired_rotation_inputs");
  auto inplace =
      function->getAttrOfType<DenseI64ArrayAttr>("intent.inplace");
  if (!version ||
      version.getValue() != "intentir.paired-last-axis-rotation.v1" ||
      !layout || layout.getValue() != "interleaved" || !inputs ||
      inputs.asArrayRef() != ArrayRef<int64_t>({0, 1, 2, 3}) || !inplace ||
      inplace.asArrayRef() != ArrayRef<int64_t>({0, 1}) ||
      function.getNumArguments() != 6 || function.getNumResults() != 0 ||
      !function.getBody().hasOneBlock())
    return std::nullopt;

  auto queryType = dyn_cast<MemRefType>(function.getArgument(0).getType());
  auto keyType = dyn_cast<MemRefType>(function.getArgument(1).getType());
  auto angleType = dyn_cast<MemRefType>(function.getArgument(2).getType());
  auto signType = dyn_cast<MemRefType>(function.getArgument(3).getType());
  auto queryOutputType =
      dyn_cast<MemRefType>(function.getArgument(4).getType());
  auto keyOutputType =
      dyn_cast<MemRefType>(function.getArgument(5).getType());
  if (!hasIdentityStaticLayout(queryType) ||
      !hasIdentityStaticLayout(keyType) ||
      !hasIdentityStaticLayout(angleType) ||
      !hasIdentityStaticLayout(signType) || queryType.getRank() != 4 ||
      keyType.getRank() != 4 || angleType.getRank() != 3 ||
      signType.getRank() != 0 || queryOutputType != queryType ||
      keyOutputType != keyType)
    return std::nullopt;

  Type elementType = queryType.getElementType();
  if (!isa<Float16Type, BFloat16Type, Float32Type>(elementType) ||
      keyType.getElementType() != elementType ||
      angleType.getElementType() != elementType ||
      !signType.getElementType().isF32())
    return std::nullopt;

  int64_t batch = queryType.getDimSize(0);
  int64_t sequence = queryType.getDimSize(1);
  int64_t queryHeads = queryType.getDimSize(2);
  int64_t headDimension = queryType.getDimSize(3);
  int64_t keyHeads = keyType.getDimSize(2);
  if (headDimension <= 0 || headDimension % 2 != 0 ||
      keyType.getDimSize(0) != batch || keyType.getDimSize(1) != sequence ||
      keyType.getDimSize(3) != headDimension ||
      angleType.getShape() !=
          ArrayRef<int64_t>({sequence, headDimension / 2, 2}))
    return std::nullopt;

  return InterleavedRotationMatch{
      function,
      function.getArgument(0),
      function.getArgument(1),
      function.getArgument(2),
      function.getArgument(3),
      function.getArgument(4),
      function.getArgument(5),
      queryType,
      keyType,
      angleType,
      batch,
      sequence,
      queryHeads,
      keyHeads,
      headDimension,
  };
}

static Value toF32(OpBuilder &builder, Location location, Value value) {
  if (value.getType().isF32())
    return value;
  return builder.create<arith::ExtFOp>(location, builder.getF32Type(), value);
}

static Value fromF32(OpBuilder &builder, Location location, Value value,
                     Type resultType) {
  if (resultType.isF32())
    return value;
  return builder.create<arith::TruncFOp>(location, resultType, value);
}

static Value vectorToF32(OpBuilder &builder, Location location, Value value,
                         VectorType floatVectorType) {
  if (cast<VectorType>(value.getType()).getElementType().isF32())
    return value;
  return builder.create<arith::ExtFOp>(location, floatVectorType, value);
}

static Value vectorFromF32(OpBuilder &builder, Location location, Value value,
                           VectorType storageVectorType) {
  if (storageVectorType.getElementType().isF32())
    return value;
  return builder.create<arith::TruncFOp>(location, storageVectorType, value);
}

static void emitFixedPacketRotation(
    OpBuilder &builder, Location location, Value source, Value output,
    Type storageType, Value batch, Value sequence, Value head, Value pair,
    Value flatAngles, Value imaginarySign, unsigned vectorWidth) {
  int64_t pairWidth = static_cast<int64_t>(vectorWidth);
  int64_t packetWidth = 2 * pairWidth;
  VectorType pairStorageType = VectorType::get({pairWidth}, storageType);
  VectorType packetStorageType = VectorType::get({packetWidth}, storageType);
  VectorType pairFloatType =
      VectorType::get({pairWidth}, builder.getF32Type());
  Value two = builder.create<arith::ConstantIndexOp>(location, 2);
  Value packetBase = builder.create<arith::MulIOp>(location, pair, two);
  Value packet = builder.create<vector::LoadOp>(
      location, packetStorageType, source,
      ValueRange{batch, sequence, head, packetBase});
  Value anglePacket = builder.create<vector::LoadOp>(
      location, packetStorageType, flatAngles,
      ValueRange{sequence, packetBase});

  SmallVector<int64_t> evenMask;
  SmallVector<int64_t> oddMask;
  SmallVector<int64_t> interleaveMask;
  evenMask.reserve(pairWidth);
  oddMask.reserve(pairWidth);
  interleaveMask.reserve(packetWidth);
  for (int64_t lane = 0; lane < pairWidth; ++lane) {
    evenMask.push_back(2 * lane);
    oddMask.push_back(2 * lane + 1);
    interleaveMask.push_back(lane);
    interleaveMask.push_back(pairWidth + lane);
  }
  Value even = builder.create<vector::ShuffleOp>(
      location, pairStorageType, packet, packet, evenMask);
  Value odd = builder.create<vector::ShuffleOp>(
      location, pairStorageType, packet, packet, oddMask);
  Value cosine = builder.create<vector::ShuffleOp>(
      location, pairStorageType, anglePacket, anglePacket, evenMask);
  Value sine = builder.create<vector::ShuffleOp>(
      location, pairStorageType, anglePacket, anglePacket, oddMask);

  Value evenF32 = vectorToF32(builder, location, even, pairFloatType);
  Value oddF32 = vectorToF32(builder, location, odd, pairFloatType);
  Value cosineF32 = vectorToF32(builder, location, cosine, pairFloatType);
  Value sineF32 = vectorToF32(builder, location, sine, pairFloatType);
  Value signVector =
      builder.create<vector::SplatOp>(location, pairFloatType, imaginarySign);
  Value scaledSine =
      builder.create<arith::MulFOp>(location, sineF32, signVector);
  Value low = builder.create<arith::SubFOp>(
      location,
      builder.create<arith::MulFOp>(location, evenF32, cosineF32),
      builder.create<arith::MulFOp>(location, oddF32, scaledSine));
  Value roundedHighCosine =
      builder.create<arith::MulFOp>(location, odd, cosine);
  Value high = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, evenF32, scaledSine),
      vectorToF32(builder, location, roundedHighCosine, pairFloatType));
  Value lowStorage =
      vectorFromF32(builder, location, low, pairStorageType);
  Value highStorage =
      vectorFromF32(builder, location, high, pairStorageType);
  Value outputPacket = builder.create<vector::ShuffleOp>(
      location, packetStorageType, lowStorage, highStorage, interleaveMask);
  builder.create<vector::StoreOp>(
      location, outputPacket, output,
      ValueRange{batch, sequence, head, packetBase});
}

static void emitRotation(OpBuilder &builder, Location location, Value source,
                         Value output, Type storageType, Value batch,
                         Value sequence, Value head, Value pair, Value cosine,
                         Value scaledSine) {
  Value two = builder.create<arith::ConstantIndexOp>(location, 2);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value evenIndex = builder.create<arith::MulIOp>(location, pair, two);
  Value oddIndex = builder.create<arith::AddIOp>(location, evenIndex, one);
  Value even = builder.create<memref::LoadOp>(
      location, source, ValueRange{batch, sequence, head, evenIndex});
  Value odd = builder.create<memref::LoadOp>(
      location, source, ValueRange{batch, sequence, head, oddIndex});

  Value evenF32 = toF32(builder, location, even);
  Value oddF32 = toF32(builder, location, odd);
  Value cosineF32 = toF32(builder, location, cosine);
  Value lowCosine =
      builder.create<arith::MulFOp>(location, evenF32, cosineF32);
  Value highSine =
      builder.create<arith::MulFOp>(location, oddF32, scaledSine);
  Value low = builder.create<arith::SubFOp>(location, lowCosine, highSine);

  // Preserve the accepted J's asymmetric evaluation order: odd*cos is first
  // rounded in storage precision and only then widened for the final f32 add.
  Value roundedHighCosine =
      builder.create<arith::MulFOp>(location, odd, cosine);
  Value highCosineF32 = toF32(builder, location, roundedHighCosine);
  Value lowSine =
      builder.create<arith::MulFOp>(location, evenF32, scaledSine);
  Value high = builder.create<arith::AddFOp>(location, lowSine, highCosineF32);

  builder.create<memref::StoreOp>(
      location, fromF32(builder, location, low, storageType), output,
      ValueRange{batch, sequence, head, evenIndex});
  builder.create<memref::StoreOp>(
      location, fromF32(builder, location, high, storageType), output,
      ValueRange{batch, sequence, head, oddIndex});
}

static void emitVectorRotation(
    OpBuilder &builder, Location location, Value source, Value output,
    Type storageType, Value batch, Value sequence, Value head, Value pair,
    Value flatAngles, Value imaginarySign, unsigned vectorWidth,
    bool scalableVector, int64_t pairCount) {
  if (!scalableVector) {
    emitFixedPacketRotation(builder, location, source, output, storageType,
                            batch, sequence, head, pair, flatAngles,
                            imaginarySign, vectorWidth);
    return;
  }
  SmallVector<bool> scalableDims{scalableVector};
  VectorType indexVectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getIndexType(), scalableDims);
  VectorType maskVectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getI1Type(), scalableDims);
  VectorType storageVectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, storageType, scalableDims);
  VectorType floatVectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getF32Type(), scalableDims);

  Value zeroIndex = builder.create<arith::ConstantIndexOp>(location, 0);
  Value oneIndex = builder.create<arith::ConstantIndexOp>(location, 1);
  Value twoIndex = builder.create<arith::ConstantIndexOp>(location, 2);
  Value bound = builder.create<arith::ConstantIndexOp>(location, pairCount);
  Value remaining = builder.create<arith::SubIOp>(location, bound, pair);
  Value mask = builder.create<vector::CreateMaskOp>(
      location, maskVectorType, ValueRange{remaining});
  Value lane = builder.create<vector::StepOp>(location, indexVectorType);
  Value pairSplat =
      builder.create<vector::SplatOp>(location, indexVectorType, pair);
  Value pairIndices = builder.create<arith::AddIOp>(location, pairSplat, lane);
  Value twoSplat =
      builder.create<vector::SplatOp>(location, indexVectorType, twoIndex);
  Value oneSplat =
      builder.create<vector::SplatOp>(location, indexVectorType, oneIndex);
  Value evenIndices =
      builder.create<arith::MulIOp>(location, pairIndices, twoSplat);
  Value oddIndices =
      builder.create<arith::AddIOp>(location, evenIndices, oneSplat);

  Value zeroStorage = builder.create<arith::ConstantOp>(
      location, storageType, builder.getZeroAttr(storageType));
  Value passThrough = builder.create<vector::SplatOp>(
      location, storageVectorType, zeroStorage);
  Value even = builder.create<vector::GatherOp>(
      location, storageVectorType, source,
      ValueRange{batch, sequence, head, zeroIndex}, evenIndices, mask,
      passThrough);
  Value odd = builder.create<vector::GatherOp>(
      location, storageVectorType, source,
      ValueRange{batch, sequence, head, zeroIndex}, oddIndices, mask,
      passThrough);
  Value cosine = builder.create<vector::GatherOp>(
      location, storageVectorType, flatAngles,
      ValueRange{sequence, zeroIndex}, evenIndices, mask,
      passThrough);
  Value sine = builder.create<vector::GatherOp>(
      location, storageVectorType, flatAngles,
      ValueRange{sequence, zeroIndex}, oddIndices, mask,
      passThrough);

  Value evenF32 = vectorToF32(builder, location, even, floatVectorType);
  Value oddF32 = vectorToF32(builder, location, odd, floatVectorType);
  Value cosineF32 = vectorToF32(builder, location, cosine, floatVectorType);
  Value sineF32 = vectorToF32(builder, location, sine, floatVectorType);
  Value signVector = builder.create<vector::SplatOp>(
      location, floatVectorType, imaginarySign);
  Value scaledSine =
      builder.create<arith::MulFOp>(location, sineF32, signVector);
  Value low = builder.create<arith::SubFOp>(
      location,
      builder.create<arith::MulFOp>(location, evenF32, cosineF32),
      builder.create<arith::MulFOp>(location, oddF32, scaledSine));
  Value roundedHighCosine =
      builder.create<arith::MulFOp>(location, odd, cosine);
  Value high = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, evenF32, scaledSine),
      vectorToF32(builder, location, roundedHighCosine, floatVectorType));
  builder.create<vector::ScatterOp>(
      location, output, ValueRange{batch, sequence, head, zeroIndex},
      evenIndices, mask,
      vectorFromF32(builder, location, low, storageVectorType));
  builder.create<vector::ScatterOp>(
      location, output, ValueRange{batch, sequence, head, zeroIndex}, oddIndices,
      mask, vectorFromF32(builder, location, high, storageVectorType));
}

template <typename Body>
static void emitFor(OpBuilder &builder, Location location, int64_t upper,
                    Body &&body) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value bound = builder.create<arith::ConstantIndexOp>(location, upper);
  auto loop = builder.create<scf::ForOp>(location, zero, bound, one);
  OpBuilder nested = OpBuilder::atBlockBegin(loop.getBody());
  body(nested, loop.getInductionVar());
}

static std::pair<Value, Value>
loadAngles(OpBuilder &builder, Location location,
           const InterleavedRotationMatch &match, Value sequence, Value pair,
           Value imaginarySign) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value cosine = builder.create<memref::LoadOp>(
      location, match.angles, ValueRange{sequence, pair, zero});
  Value sine = builder.create<memref::LoadOp>(
      location, match.angles, ValueRange{sequence, pair, one});
  Value scaledSine = builder.create<arith::MulFOp>(
      location, toF32(builder, location, sine), imaginarySign);
  return {cosine, scaledSine};
}

static void lowerContiguous(const InterleavedRotationMatch &match,
                            OpBuilder &builder, Location location,
                            Value imaginarySign, unsigned vectorWidth,
                            bool scalableVector) {
  Value flatAngles;
  if (vectorWidth != 1) {
    auto flatAngleType = MemRefType::get(
        {match.sequence, match.headDimension},
        match.angleType.getElementType());
    SmallVector<ReassociationIndices> reassociation{{0}, {1, 2}};
    flatAngles = builder.create<memref::CollapseShapeOp>(
        location, flatAngleType, match.angles, reassociation);
  }
  auto emitTensor = [&](Value source, Value output, int64_t heads) {
    emitFor(builder, location, match.batch, [&](OpBuilder &bb, Value batch) {
      emitFor(bb, location, match.sequence, [&](OpBuilder &sb, Value sequence) {
        emitFor(sb, location, heads, [&](OpBuilder &hb, Value head) {
          if (vectorWidth == 1) {
            emitFor(hb, location, match.headDimension / 2,
                    [&](OpBuilder &pb, Value pair) {
                      auto [cosine, scaledSine] = loadAngles(
                          pb, location, match, sequence, pair, imaginarySign);
                      emitRotation(pb, location, source, output,
                                   match.queryType.getElementType(), batch,
                                   sequence, head, pair, cosine, scaledSine);
                    });
          } else {
            Value zero = hb.create<arith::ConstantIndexOp>(location, 0);
            Value bound = hb.create<arith::ConstantIndexOp>(
                location, match.headDimension / 2);
            Value step = hb.create<arith::ConstantIndexOp>(location, vectorWidth);
            if (scalableVector) {
              Value vscale = hb.create<vector::VectorScaleOp>(location);
              step = hb.create<arith::MulIOp>(location, step, vscale);
            }
            auto pairs = hb.create<scf::ForOp>(location, zero, bound, step);
            OpBuilder pb = OpBuilder::atBlockBegin(pairs.getBody());
            emitVectorRotation(
                pb, location, source, output, match.queryType.getElementType(),
                batch, sequence, head, pairs.getInductionVar(), flatAngles,
                imaginarySign, vectorWidth, scalableVector,
                match.headDimension / 2);
          }
        });
      });
    });
  };
  emitTensor(match.query, match.queryOutput, match.queryHeads);
  emitTensor(match.key, match.keyOutput, match.keyHeads);
}

static void lowerHeadAngleReuse(const InterleavedRotationMatch &match,
                                OpBuilder &builder, Location location,
                                Value imaginarySign) {
  emitFor(builder, location, match.batch, [&](OpBuilder &bb, Value batch) {
    emitFor(bb, location, match.sequence, [&](OpBuilder &sb, Value sequence) {
      emitFor(sb, location, match.headDimension / 2,
              [&](OpBuilder &pb, Value pair) {
                auto [cosine, scaledSine] = loadAngles(
                    pb, location, match, sequence, pair, imaginarySign);
                emitFor(pb, location, match.queryHeads,
                        [&](OpBuilder &hb, Value head) {
                          emitRotation(hb, location, match.query,
                                       match.queryOutput,
                                       match.queryType.getElementType(), batch,
                                       sequence, head, pair, cosine, scaledSine);
                        });
                emitFor(pb, location, match.keyHeads,
                        [&](OpBuilder &hb, Value head) {
                          emitRotation(hb, location, match.key, match.keyOutput,
                                       match.keyType.getElementType(), batch,
                                       sequence, head, pair, cosine, scaledSine);
                        });
              });
    });
  });
}

static void lowerInterleavedRotation(InterleavedRotationMatch match,
                                     bool headAngleReuse, unsigned vectorWidth,
                                     bool scalableVector) {
  Block &body = match.function.getBody().front();
  while (!body.empty())
    body.back().erase();
  OpBuilder builder(&body, body.end());
  Location location = match.function.getLoc();
  Value imaginarySign = builder.create<memref::LoadOp>(
      location, match.imaginarySign, ValueRange{});
  if (headAngleReuse)
    lowerHeadAngleReuse(match, builder, location, imaginarySign);
  else
    lowerContiguous(match, builder, location, imaginarySign, vectorWidth,
                    scalableVector);
  builder.create<func::ReturnOp>(location);
}

struct LowerPairedRotationToLoopsPass
    : ::intent::impl::LowerPairedRotationToLoopsBase<
          LowerPairedRotationToLoopsPass> {
  using LowerPairedRotationToLoopsBase::LowerPairedRotationToLoopsBase;

  void runOnOperation() override {
    if (vectorWidth != 1 && vectorWidth != 2 && vectorWidth != 4 &&
        vectorWidth != 8 && vectorWidth != 16) {
      getOperation().emitError(
          "paired-rotation vector width must be 1, 2, 4, 8, or 16");
      signalPassFailure();
      return;
    }
    if (scalableVector && vectorWidth == 1) {
      getOperation().emitError(
          "scalable paired rotation requires a vector width greater than one");
      signalPassFailure();
      return;
    }
    if (headAngleReuse && vectorWidth != 1) {
      getOperation().emitError(
          "head-angle reuse currently requires the scalar pair traversal");
      signalPassFailure();
      return;
    }
    SmallVector<InterleavedRotationMatch> matches;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      if (auto match = matchInterleavedRotation(function))
        matches.push_back(*match);
    if (matches.empty()) {
      if (requireLowering) {
        getOperation().emitError(
            "required closed interleaved paired rotation was not present");
        signalPassFailure();
      }
      return;
    }
    if (!scalableVector && vectorWidth > 1 &&
        llvm::any_of(matches, [&](const InterleavedRotationMatch &match) {
          return (match.headDimension / 2) % vectorWidth != 0;
        })) {
      getOperation().emitError(
          "fixed packet paired rotation requires vector width to divide the "
          "number of rotary pairs");
      signalPassFailure();
      return;
    }
    for (InterleavedRotationMatch match : matches)
      lowerInterleavedRotation(match, headAngleReuse, vectorWidth,
                               scalableVector);
    Builder builder(getOperation().getContext());
    getOperation()->setAttr("intent.rvv.paired_rotation_lowering_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.paired_rotation_head_angle_reuse_count",
                            builder.getI64IntegerAttr(
                                headAngleReuse ? matches.size() : 0));
    getOperation()->setAttr("intent.rvv.paired_rotation_vector_width",
                            builder.getI64IntegerAttr(vectorWidth));
    getOperation()->setAttr("intent.rvv.paired_rotation_scalable_count",
                            builder.getI64IntegerAttr(
                                scalableVector ? matches.size() : 0));
  }
};

} // namespace
