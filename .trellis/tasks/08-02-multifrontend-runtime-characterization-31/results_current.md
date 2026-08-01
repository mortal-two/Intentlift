# Task31 multi-frontend runtime and artifact characterization

- denominator: 31 same-origin pairs / 62 accepted frontend artifacts
- Q: exact Task28 Gate2 input digest reconstructed for every artifact
- runtime edge: recompiled Task30 CUDA cubin graph versus canonical RefJ
- Source / Replay / pi / explicit frontend label are unavailable to runtime_consume.py

## Runtime closure

| Evidence | Result |
|---|---:|
| Exact accepted Gate2 Q reconstructed | 62/62 |
| RefJ execution | 62/62 |
| CUDA target vs RefJ agreement | 62/62 |
| Recompiled cubin digests identical to Task30 | 62/62 |
| Prepared graph benchmark available | 62/62 |
| Both frontends runtime-close per pair | 31/31 |

## Per-frontend backend/runtime characterization

| Frontend | N | Serial compile median / p90 (ms) | Kernels median | Cubin bytes median | Prepared graph median / p90 (us) | RVV object median (B) |
|---|---:|---:|---:|---:|---:|---:|
| tilelang | 31 | 314.602 / 561.988 | 4.000 | 23744.0 | 17.154 / 91.757 | 4592.0 |
| cuda | 31 | 296.758 / 514.368 | 6.000 | 34784.0 | 20.653 / 64.720 | 4592.0 |

Compile time is serial compiler-pipeline wall time; prepared-graph latency uses `triton.testing.do_bench` on an allocation-free/no-sync whole graph. Artifacts run in independent processes without pinned GPU clocks, so timings are descriptive distributions only: neither a Source/Replay baseline nor a paired frontend speedup.

## Pair stability by J correspondence

| Stratum | Pairs | Runtime both | CUDA kernel-count equal | RVV instruction-count equal | Median cubin-size rel. delta | Median RVV-object rel. delta |
|---|---:|---:|---:|---:|---:|---:|
| Semantic core exact | 26 | 26/26 | 26/26 | 26/26 | 0.000 | 0.000 |
| Semantic core non-exact | 5 | 5/5 | 0/5 | 4/5 | 0.244 | 0.091 |

## Cross-frontend generated-artifact identity

| Stratum | Pairs | Intent MLIR digest equal | CUDA cubin sequence equal | RVV object digest equal | Runtime graph topology equal | Output signature equal |
|---|---:|---:|---:|---:|---:|---:|
| Semantic core exact | 26 | 26/26 | 26/26 | 26/26 | 26/26 | 26/26 |
| Semantic core non-exact | 5 | 0/5 | 0/5 | 0/5 | 0/5 | 4/5 |

## Numeric agreement receipts

- output comparisons: 82/82 pass

| Policy | Cases | Output comparisons | Exact outputs | Tolerance outputs | Max observed absolute error |
|---|---:|---:|---:|---:|---:|
| `intentir.gate2.default.v1` | 52 | 72 | 8 | 64 | 0.000001907 |
| `intentir.gate2.tf32-tensorcore.v1` | 10 | 10 | 0 | 10 | 0.000000238 |

## Artifact diversity

- CUDA cubin records: 401; unique digests: 102
- RVV objects: 62; unique digests: 36

## Failure taxonomy

- none

## Per-pair runtime status

| Stem | Core exact | Runtime both | Task30 recompile stable | Cross-frontend target artifacts equal | CUDA kernels equal | RVV instructions equal |
|---|---|---|---|---|---|---|
| `_attn_fwd` | yes | yes | yes | yes | yes | yes |
| `add2d` | yes | yes | yes | yes | yes | yes |
| `add_bias2d` | yes | yes | yes | yes | yes | yes |
| `any_kernel_dim` | no | yes | yes | no | no | yes |
| `clamp2d` | yes | yes | yes | yes | yes | yes |
| `copy2d_divmod` | yes | yes | yes | yes | yes | yes |
| `exp2d` | yes | yes | yes | yes | yes | yes |
| `flash_attention2d` | yes | yes | yes | yes | yes | yes |
| `floor2d` | yes | yes | yes | yes | yes | yes |
| `gather2d` | no | yes | yes | no | no | no |
| `gemm` | yes | yes | yes | yes | yes | yes |
| `grouped_row_sum2d` | yes | yes | yes | yes | yes | yes |
| `groupnorm` | yes | yes | yes | yes | yes | yes |
| `layer_norm_residual2d` | yes | yes | yes | yes | yes | yes |
| `layernorm` | yes | yes | yes | yes | yes | yes |
| `masked_attention2d` | yes | yes | yes | yes | yes | yes |
| `masked_softmax2d` | no | yes | yes | no | no | yes |
| `matmul_bias_relu2d` | yes | yes | yes | yes | yes | yes |
| `matmul_fused_epilogue2d` | no | yes | yes | no | no | yes |
| `matmul_relu2d` | yes | yes | yes | yes | yes | yes |
| `mlp2d` | yes | yes | yes | yes | yes | yes |
| `relu2d` | yes | yes | yes | yes | yes | yes |
| `rms_norm2d` | yes | yes | yes | yes | yes | yes |
| `rms_norm_residual2d` | yes | yes | yes | yes | yes | yes |
| `row_max` | yes | yes | yes | yes | yes | yes |
| `row_sum` | yes | yes | yes | yes | yes | yes |
| `rowmask_where2d` | no | yes | yes | no | no | yes |
| `softmax_inner` | yes | yes | yes | yes | yes | yes |
| `transpose2d` | yes | yes | yes | yes | yes | yes |
| `upsample_bicubic2d_aa` | yes | yes | yes | yes | yes | yes |
| `where2d` | yes | yes | yes | yes | yes | yes |
