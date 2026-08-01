//===- FlashMMAFragments.h - shared Flash MMA fragment loaders -*- C++ -*-===//
//
// Internal helpers shared by the FlashAttention forward and backward
// lowerings.  They encode the already-validated ldmatrix.x4 lane mapping used
// to fetch two adjacent m16n8k16 B fragments with one instruction.  Keeping
// this mapping here prevents the two fused kernels from drifting apart.
//
//===----------------------------------------------------------------------===//

#ifndef INTENT_LIB_CONVERSION_CUDA_FLASHMMAFRAGMENTS_H
#define INTENT_LIB_CONVERSION_CUDA_FLASHMMAFRAGMENTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::cuda_detail {

using namespace mlir;

/// Load two adjacent non-transposed B fragments from a source whose final two
/// dimensions are [n, k].  `leadingIndices` names any batch/stage dimensions.
inline Value loadMmaBPair(OpBuilder &builder, Location loc, Value source,
                          Value nBase, Value kBase, Value laneDiv8,
                          Value laneMod8, Value constant2, Value constant8,
                          VectorType pairType,
                          ValueRange leadingIndices = ValueRange()) {
  Value pairRowGroup =
      builder.create<arith::DivUIOp>(loc, laneDiv8, constant2);
  Value pairRowOffset =
      builder.create<arith::MulIOp>(loc, pairRowGroup, constant8);
  Value row = builder.create<arith::AddIOp>(
      loc, nBase,
      builder.create<arith::AddIOp>(loc, pairRowOffset, laneMod8));
  Value pairColumnGroup =
      builder.create<arith::RemUIOp>(loc, laneDiv8, constant2);
  Value pairColumnOffset =
      builder.create<arith::MulIOp>(loc, pairColumnGroup, constant8);
  Value column = builder.create<arith::AddIOp>(loc, kBase, pairColumnOffset);
  SmallVector<Value> indices(leadingIndices.begin(), leadingIndices.end());
  indices.push_back(row);
  indices.push_back(column);
  return builder.create<nvgpu::LdMatrixOp>(
      loc, pairType, source, indices, /*transpose=*/false, /*numTiles=*/4);
}

/// Load two adjacent transposed B fragments from a source whose final two
/// dimensions are [k, n].  `leadingIndices` names any batch/stage dimensions.
inline Value loadMmaBTransposedPair(
    OpBuilder &builder, Location loc, Value source, Value nBase, Value kBase,
    Value laneMod16, Value laneDiv16, Value constant8, VectorType pairType,
    ValueRange leadingIndices = ValueRange()) {
  Value row = builder.create<arith::AddIOp>(loc, kBase, laneMod16);
  Value pairColumnOffset =
      builder.create<arith::MulIOp>(loc, laneDiv16, constant8);
  Value column = builder.create<arith::AddIOp>(loc, nBase, pairColumnOffset);
  SmallVector<Value> indices(leadingIndices.begin(), leadingIndices.end());
  indices.push_back(row);
  indices.push_back(column);
  return builder.create<nvgpu::LdMatrixOp>(
      loc, pairType, source, indices, /*transpose=*/true, /*numTiles=*/4);
}

/// Split one ldmatrix.x4 result into either of its two m16n8k16 B operands.
/// This is an SSA-only operation: it emits vector extracts/inserts but no
/// additional memory transaction.
inline Value extractMmaBPairHalf(OpBuilder &builder, Location loc, Value pair,
                                 Value zeroFragment, int64_t half) {
  Value fragment = zeroFragment;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t column = 0; column < 2; ++column) {
      Value item = builder.create<vector::ExtractOp>(
          loc, pair, ArrayRef<int64_t>{half * 2 + row, column});
      fragment = builder.create<vector::InsertOp>(
          loc, item, fragment, ArrayRef<int64_t>{row, column});
    }
  }
  return fragment;
}

} // namespace intent::cuda_detail

#endif // INTENT_LIB_CONVERSION_CUDA_FLASHMMAFRAGMENTS_H
