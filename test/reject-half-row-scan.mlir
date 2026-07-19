// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=32 vector-width=4' %s | FileCheck %s

// The scan rewriter currently owns an f32-only register/shared-memory ABI.
// Explicit mixed-precision graphs must remain untouched even when their row
// geometry would otherwise qualify for vec4; treating a bf16 pointer as
// vector<4xf32> would double the packet width and could write out of bounds.

// CHECK-LABEL: func.func @bf16_scan_fail_closed(
// CHECK-NOT:     gpu.launch
// CHECK:         intent.scan
// CHECK:         intent.cast
func.func @bf16_scan_fail_closed(%x: tensor<2x128xbf16>)
    -> tensor<2x128xbf16> {
  %xf = intent.cast %x {to_dtype = "f32"}
    : (tensor<2x128xbf16>) -> tensor<2x128xf32>
  %scan = intent.scan %xf {fn = "sum", axis = 1}
    : (tensor<2x128xf32>) -> tensor<2x128xf32>
  %y = intent.cast %scan {to_dtype = "bf16"}
    : (tensor<2x128xf32>) -> tensor<2x128xbf16>
  return %y : tensor<2x128xbf16>
}
