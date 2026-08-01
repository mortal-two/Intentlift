# Task28 current results

- accepted unique pairs: 26/31
- accepted unique frontend runs: 52/62
- counting: 70 cohort memberships / 31 unique pairs / 62 unique frontend runs

| Cohort | Accepted | Complete |
|---|---:|---|
| `g1_pointwise_selection` | 10/10 | yes |
| `g2_reduction_statistics_normalization` | 10/10 | yes |
| `g3_contraction_epilogue` | 5/10 | no |
| `g4_attention_masking` | 10/10 | yes |
| `g5_indexing_layout_image` | 10/10 | yes |
| `g6_fusion_composition` | 6/10 | no |
| `g7_cross_feature_stress` | 8/10 | no |

| Stem | TileLang | CUDA | Pair |
|---|---|---|---|
| `_attn_fwd` | PASS | PASS | PASS |
| `add2d` | PASS | PASS | PASS |
| `add_bias2d` | PASS | PASS | PASS |
| `any_kernel_dim` | PASS | PASS | PASS |
| `clamp2d` | PASS | PASS | PASS |
| `copy2d_divmod` | PASS | PASS | PASS |
| `exp2d` | PASS | PASS | PASS |
| `flash_attention2d` | PASS | PASS | PASS |
| `floor2d` | PASS | PASS | PASS |
| `gather2d` | PASS | PASS | PASS |
| `gemm` | — | — | — |
| `grouped_row_sum2d` | PASS | PASS | PASS |
| `groupnorm` | PASS | PASS | PASS |
| `layer_norm_residual2d` | PASS | PASS | PASS |
| `layernorm` | PASS | PASS | PASS |
| `masked_attention2d` | PASS | PASS | PASS |
| `masked_softmax2d` | PASS | PASS | PASS |
| `matmul_bias_relu2d` | — | — | — |
| `matmul_fused_epilogue2d` | — | — | — |
| `matmul_relu2d` | — | — | — |
| `mlp2d` | — | — | — |
| `relu2d` | PASS | PASS | PASS |
| `rms_norm2d` | PASS | PASS | PASS |
| `rms_norm_residual2d` | PASS | PASS | PASS |
| `row_max` | PASS | PASS | PASS |
| `row_sum` | PASS | PASS | PASS |
| `rowmask_where2d` | PASS | PASS | PASS |
| `softmax_inner` | PASS | PASS | PASS |
| `transpose2d` | PASS | PASS | PASS |
| `upsample_bicubic2d_aa` | PASS | PASS | PASS |
| `where2d` | PASS | PASS | PASS |
