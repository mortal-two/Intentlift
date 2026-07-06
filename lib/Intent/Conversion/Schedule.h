//===- Schedule.h - first-class GPU schedule + structured launch plan ----===//
//
// M6 (R5 P3/P5): a first-class `Schedule` struct capturing the *schedule knobs*
// (grid mapping, block dims, per-thread tile, vector width, smem plan, pipeline
// depth, split factor) SEPARATE from an op's math. Today each ~500-line
// `ConvertIntent*ToGPU` pass welds the schedule into its `rewrite()` body and
// bakes the launch geometry as raw `arith.constant`s that the Python runtime
// then recovers by REGEX-scraping the fully-lowered LLVM text
// (`orchestrate.py`/`row_reduction.py`). That is brittle by construction.
//
// This header gives two things, both ADDITIVE (behavior-preserving):
//   1. `Schedule` — the struct the M5 autotuner's config dict maps onto, and the
//      single carrier of a pass's fork decision (register-tile vs tiled-smem vs
//      scan; WMMA vs mma.sync; one-block-per-row vs flat-grid-stride).
//   2. `attachLaunchPlan()` — emits the pass's chosen grid/block/smem as a
//      STRUCTURED `intent.launch_plan` dictionary attribute on the host func, so
//      the runtime can read the launch config from structured data instead of
//      regex-walking `%ctaid`/`.reqntid` predicates. The runtime PREFERS this
//      plan and ASSERTs it equals the regex-recovered values (equivalence proof),
//      then trusts it; the regex path stays as a validated fallback.
//
//===----------------------------------------------------------------------===//

#ifndef INTENT_CONVERSION_SCHEDULE_H
#define INTENT_CONVERSION_SCHEDULE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {

/// How the grid maps onto the output space.
///   OneBlockPerRow  — the legacy row-cooperative launch (grid = M).
///   FlatGridStride  — a numSM-sized block-stride grid over the flat element
///                     space (the M1 occupancy fix, tail-mask for free).
enum class GridStyle { OneBlockPerRow, FlatGridStride };

/// A first-class GPU schedule: the tuning knobs, separate from the op's math.
/// One parametric codegen consumes `(op-DAG, Schedule) -> gpu.launch`; a pass's
/// fork decision (which specialized emit body to run) is carried in `kind`.
struct Schedule {
  // Launch geometry (grid blocks / block threads, x-y-z).
  int64_t gridX = 1, gridY = 1, gridZ = 1;
  int64_t blockX = 1, blockY = 1, blockZ = 1;
  // Per-thread register/feature tile (K columns owned by one thread, or the
  // tiled Kpt strip); 1 for a scalar-per-thread schedule.
  int64_t perThreadTile = 1;
  // 128-bit vector width: 1 = scalar ld/st, 4 = vector<4xf32> coalesced.
  int64_t vectorWidth = 1;
  // Dynamic shared-memory bytes requested at launch (0 = none / static only).
  int64_t smemBytes = 0;
  // cp.async / mma software-pipeline depth (num-stages); 1 = unpipelined.
  int64_t pipelineDepth = 1;
  // split-K (contraction) / split-row (segmented reduce) factor; 1 = none.
  int64_t splitFactor = 1;
  GridStyle gridStyle = GridStyle::OneBlockPerRow;
  // Pass-local schedule selector (which specialized emit body). Interpreted by
  // the owning pass; carried here so the fork decision lives on the Schedule.
  int kind = 0;
};

/// Attach `s`'s launch geometry to `hostFunc` as a structured, discardable
/// `intent.launch_plan` dictionary attribute the runtime reads instead of
/// regex-scraping lowered LLVM. Fields are plain i64 so the Python side parses
/// them trivially. Purely additive: unknown discardable attrs are ignored by
/// every downstream pass, so this cannot change any launch's behavior.
inline void attachLaunchPlan(mlir::Operation *hostFunc, const Schedule &s) {
  using namespace mlir;
  MLIRContext *ctx = hostFunc->getContext();
  Builder bld(ctx);
  auto arr3 = [&](int64_t a, int64_t b, int64_t c) {
    return bld.getI64ArrayAttr({a, b, c});
  };
  SmallVector<NamedAttribute, 8> f;
  f.push_back(bld.getNamedAttr("grid", arr3(s.gridX, s.gridY, s.gridZ)));
  f.push_back(bld.getNamedAttr("block", arr3(s.blockX, s.blockY, s.blockZ)));
  f.push_back(bld.getNamedAttr("smem", bld.getI64IntegerAttr(s.smemBytes)));
  f.push_back(
      bld.getNamedAttr("vector_width", bld.getI64IntegerAttr(s.vectorWidth)));
  f.push_back(bld.getNamedAttr("per_thread_tile",
                               bld.getI64IntegerAttr(s.perThreadTile)));
  f.push_back(bld.getNamedAttr("pipeline_depth",
                               bld.getI64IntegerAttr(s.pipelineDepth)));
  f.push_back(
      bld.getNamedAttr("split_factor", bld.getI64IntegerAttr(s.splitFactor)));
  hostFunc->setAttr("intent.launch_plan", bld.getDictionaryAttr(f));
}

} // namespace intent

#endif // INTENT_CONVERSION_SCHEDULE_H
