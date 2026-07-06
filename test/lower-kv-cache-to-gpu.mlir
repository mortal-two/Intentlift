// RUN: intent-opt -convert-intent-gather-to-gpu=block-threads=4 %s | FileCheck %s

// KV-cache paged write (reshape_and_cache_flash / concat_and_cache_mla) lowered
// by the SAME C3 `convert-intent-gather-to-gpu` pass as gather/scatter — NO
// per-kernel branch. The paged cache flattens to [num_blocks*block_size, D] and
// the write is cache[slot[t], :] = kv[t, :], expressed as an intent.scatter
// ALONG AXIS 0: the slot index (broadcast to [Nt, D]) is the data-dependent ROW
// address, so the store coordinate on `data` is data[idx, col] = src[t, col].
// Scatter is IN-PLACE on the cache (no appended output arg). Nt=4 tokens, D=8,
// cache rows Rc=6, T=4.
// CHECK-LABEL: func.func @kv_cache(
// CHECK-SAME:    %[[DATA:.*]]: memref<6x8xf32>, %[[IDX:.*]]: memref<4x8xi64>, %[[SRC:.*]]: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// src loaded at [bx, col]; slot index loaded at [bx, col] and index_cast:
// CHECK:         memref.load %[[SRC]][%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         %[[RAW:.*]] = memref.load %[[IDX]][%{{.*}}, %{{.*}}] : memref<4x8xi64>
// CHECK:         %[[SLOT:.*]] = arith.index_cast %[[RAW]] : i64 to index
// data-dependent store on the paged cache ROW = slot (axis 0):
// CHECK:         memref.store %{{.*}}, %[[DATA]][%[[SLOT]], %{{.*}}] : memref<6x8xf32>
// CHECK:         gpu.terminator
func.func @kv_cache(%data: tensor<6x8xf32>, %index: tensor<4x8xi64>, %src: tensor<4x8xf32>)
    -> tensor<6x8xf32> {
  %o = intent.scatter %data, %index, %src {axis = 0 : i64, mode = "along_axis", reduce = "none"}
      : (tensor<6x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<6x8xf32>
  return %o : tensor<6x8xf32>
}
