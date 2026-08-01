# Task30 artifact-only multi-frontend backend reuse results

- denominator: 31 same-origin pairs / 62 exact accepted frontend artifacts
- backend input: anonymous content-addressed `ProductionSnapshot(J,C)` plus canonical logical extents
- backend-visible Source / pi / Replay / explicit frontend label / case-id / Gate evidence / historical winner: none
- Primary-20 is not used as an experimental denominator

## Artifact-level reuse matrix

| Frontend | N | Decode | Emit Intent MLIR | CUDA Legal | CUDA cubin | RVV Legal | RVV object |
|---|---:|---:|---:|---:|---:|---:|---:|
| tilelang | 31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 |
| cuda | 31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 |

## Pair-level both-frontends matrix

| Stage | Both TileLang and CUDA |
|---|---:|
| Decode | 31/31 |
| Emit Intent MLIR | 31/31 |
| CUDA Contract Legal | 31/31 |
| CUDA >=1 real lower state | 31/31 |
| RVV Contract Legal | 31/31 |
| RVV >=1 real lower state | 31/31 |

## Joint frontend evidence

| Evidence | Full 31 |
|---|---:|
| Frontend closure | 31/31 pairs; 62/62 artifacts |
| Signature-free semantic core exact | 26/31 |
| Op-kind set exact | 29/31 |
| Boundary exact | 26/31 |
| Carrier-normalized buffer profile exact | 27/31 |
| Raw full-J exact | 0/31 |
| C exact | 31/31 |
| CUDA has extra runtime scalar | 27/31 |
| Frontend-specific backend LoC | 0 |
| Frontend-specific/per-kernel backend rules | 0 / 0 |
| Shared J-feature consumer delta | +31 LoC / 1 file (`math.floor`, CUDA) |
| CUDA lower before -> after shared fix | 60/62 -> 62/62 |

`CUDA cubin` means the real Intent -> Linalg -> NVVM/PTX -> ptxas path produced at least one ELF cubin. `RVV object` means the real Intent -> Linalg -> LLVM -> RVV opt/llc path produced a schedule-realized native object. Neither column is a runtime-correctness or performance claim; behavioral closure remains Task28 Gate1/Gate2, and this Task does not execute Q/D or benchmark the binaries.

## Failures

- none

## Per-pair status

| Stem | Decode | Emit | CUDA Legal | CUDA Lower | RVV Legal | RVV Lower |
|---|---|---|---|---|---|---|
| `_attn_fwd` | PASS | PASS | PASS | PASS | PASS | PASS |
| `add2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `add_bias2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `any_kernel_dim` | PASS | PASS | PASS | PASS | PASS | PASS |
| `clamp2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `copy2d_divmod` | PASS | PASS | PASS | PASS | PASS | PASS |
| `exp2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `flash_attention2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `floor2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `gather2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `gemm` | PASS | PASS | PASS | PASS | PASS | PASS |
| `grouped_row_sum2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `groupnorm` | PASS | PASS | PASS | PASS | PASS | PASS |
| `layer_norm_residual2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `layernorm` | PASS | PASS | PASS | PASS | PASS | PASS |
| `masked_attention2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `masked_softmax2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `matmul_bias_relu2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `matmul_fused_epilogue2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `matmul_relu2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `mlp2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `relu2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `rms_norm2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `rms_norm_residual2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `row_max` | PASS | PASS | PASS | PASS | PASS | PASS |
| `row_sum` | PASS | PASS | PASS | PASS | PASS | PASS |
| `rowmask_where2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `softmax_inner` | PASS | PASS | PASS | PASS | PASS | PASS |
| `transpose2d` | PASS | PASS | PASS | PASS | PASS | PASS |
| `upsample_bicubic2d_aa` | PASS | PASS | PASS | PASS | PASS | PASS |
| `where2d` | PASS | PASS | PASS | PASS | PASS | PASS |
