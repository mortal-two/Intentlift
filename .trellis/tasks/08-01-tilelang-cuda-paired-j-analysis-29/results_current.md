# Task29 TileLang / generated-CUDA paired-J results

- primary detailed cohort: 20 unique pairs / 40 accepted artifacts
- full robustness cohort: 31 unique pairs / 62 accepted artifacts
- analysis calls Lift/Replay: 0
- full-J equality is descriptive, never an acceptance gate

| Metric | Primary 20 | Full 31 |
|---|---:|---:|
| Full canonical J exact | 0/20 | 0/31 |
| Identity-normalized J exact | 0/20 | 0/31 |
| Signature-free semantic core exact | 15/20 | 26/31 |
| Layer-A intent exact | 15/20 | 26/31 |
| Op-kind set exact | 18/20 | 29/31 |
| Boundary exact | 15/20 | 26/31 |
| Raw logical signature exact | 0/20 | 0/31 |
| Carrier-normalized typed buffer profile exact | 16/20 | 27/31 |
| C exact | 20/20 | 31/31 |
| Compact pi exact | 20/20 | 31/31 |

## Exact accepted Replay cost

| Cohort / frontend | Runs | Gate1 median (s) | Gate1 mean (s) | Replay bytes median | Provider calls |
|---|---:|---:|---:|---:|---:|
| Primary 20 / tilelang | 20 | 197.914 | 250.386 | 572.0 | 20 |
| Primary 20 / cuda | 20 | 125.408 | 127.758 | 729.0 | 20 |
| Full 31 / tilelang | 31 | 198.701 | 236.049 | 690.0 | 31 |
| Full 31 / cuda | 31 | 128.694 | 129.945 | 916.0 | 31 |

## Difference taxonomy

| Category | Primary 20 | Full 31 |
|---|---:|---:|
| `access` | 5 | 5 |
| `boundary` | 5 | 5 |
| `buffer_order` | 8 | 13 |
| `dimension_contract` | 1 | 1 |
| `effect` | 5 | 5 |
| `layer_a_typed_representation` | 5 | 5 |
| `runtime_scalar_abi` | 18 | 27 |
| `tensor_pointer_carrier` | 20 | 31 |
| `typed_buffer_profile` | 4 | 4 |

## Boundary field exactness

| Boundary field | Primary 20 | Full 31 |
|---|---:|---:|
| `access_guard_relations` | 20/20 | 31/31 |
| `accesses` | 15/20 | 26/31 |
| `aliases` | 20/20 | 31/31 |
| `broadcasts` | 16/20 | 27/31 |
| `consumer_extensions` | 20/20 | 31/31 |
| `domains` | 20/20 | 31/31 |
| `effects` | 15/20 | 26/31 |
| `index_bounds` | 19/20 | 30/31 |
| `output_mappings` | 20/20 | 31/31 |
| `realization_authorizations` | 20/20 | 31/31 |
| `reductions` | 19/20 | 30/31 |
| `result_relations` | 19/20 | 30/31 |
| `updates` | 20/20 | 31/31 |

## Primary pair details

| Stem | Core | Intent | Ops | Boundary | Buffer profile | Extra CUDA scalars | C | Difference categories |
|---|---|---|---|---|---|---:|---|---|
| `add2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `add_bias2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order |
| `any_kernel_dim` | no | no | no | no | no | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, typed_buffer_profile, layer_a_typed_representation, boundary, access, effect |
| `copy2d_divmod` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `exp2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `flash_attention2d` | yes | yes | yes | yes | yes | 0 | yes | tensor_pointer_carrier, buffer_order |
| `gather2d` | no | no | no | no | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order, layer_a_typed_representation, boundary, access, effect, dimension_contract |
| `gemm` | yes | yes | yes | yes | yes | 3 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `grouped_row_sum2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `groupnorm` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order |
| `layer_norm_residual2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order |
| `layernorm` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order |
| `masked_attention2d` | yes | yes | yes | yes | yes | 0 | yes | tensor_pointer_carrier, buffer_order |
| `masked_softmax2d` | no | no | yes | no | no | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, typed_buffer_profile, layer_a_typed_representation, boundary, access, effect |
| `matmul_fused_epilogue2d` | no | no | yes | no | no | 3 | yes | tensor_pointer_carrier, runtime_scalar_abi, typed_buffer_profile, layer_a_typed_representation, boundary, access, effect |
| `mlp2d` | yes | yes | yes | yes | yes | 4 | yes | tensor_pointer_carrier, runtime_scalar_abi, buffer_order |
| `row_sum` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `rowmask_where2d` | no | no | yes | no | no | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi, typed_buffer_profile, layer_a_typed_representation, boundary, access, effect |
| `transpose2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |
| `where2d` | yes | yes | yes | yes | yes | 1 | yes | tensor_pointer_carrier, runtime_scalar_abi |

All rows dereference Task28's latest accepted `selected.run_dir` and verify the run-local summary, J-only visibility receipt, Gate1 artifact digest, and passing Gate2 certificate.
