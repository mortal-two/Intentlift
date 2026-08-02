// RUN: intent-opt -convert-intent-contraction-to-gpu-mma %s | FileCheck %s --check-prefix=MIXED
// RUN: intent-opt -convert-intent-contraction-to-gpu-mma='block-m=0 block-n=64' %s | FileCheck %s --check-prefix=ZERO
// RUN: intent-opt -convert-intent-contraction-to-gpu-mma='block-m=64 block-n=64 block-k=16 warps-n=3' %s | FileCheck %s --check-prefix=PACKET

// MIXED-LABEL: func.func @mixed_storage_rejected(
// MIXED-NOT:     gpu.launch
// MIXED:         intent.contraction
func.func @mixed_storage_rejected(%A: tensor<64x32xf16>,
                                  %W: tensor<64x32xbf16>)
    -> tensor<64x64xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<64x32xf16>, tensor<64x32xbf16>) -> tensor<64x64xf32>
  return %Y : tensor<64x64xf32>
}

// ZERO-LABEL: func.func @bf16_geometry_fail_closed(
// ZERO-NOT:     gpu.launch
// ZERO:         intent.contraction
// PACKET-LABEL: func.func @bf16_geometry_fail_closed(
// PACKET-NOT:     gpu.launch
// PACKET:         intent.contraction
func.func @bf16_geometry_fail_closed(%A: tensor<64x32xbf16>,
                                     %W: tensor<64x32xbf16>)
    -> tensor<64x64xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x64xf32>
  return %Y : tensor<64x64xf32>
}
